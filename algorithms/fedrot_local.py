import copy
import torch
import numpy as np
from utils.utils import compute_accuracy
from algorithms.otloss import *
from algorithms.symmetricCE import SCELoss
from sklearn.mixture import GaussianMixture


def train_net_fedot_local(net, train_dataloader, epochs, args, device):
    """Local training function that does not rely on global class centers, only computes OT loss locally"""
    net.cuda()
    net.train()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    lambda_ot = getattr(args, 'ot_loss_weight', 1.0)
    lambda_sep = getattr(args, 'center_sep_weight', 0.05)
    num_classes = args.num_classes if hasattr(args, 'num_classes') else args.num_class
    feat_dim = None
    
    for epoch in range(epochs):
        # === Local class center accumulation statistics ===
        local_center_sum = {}
        local_center_count = {}
        for c in range(num_classes):
            local_center_sum[c] = None  # Accumulate features
            local_center_count[c] = 0   # Accumulate counts
            
        for batch in train_dataloader:
            if len(batch) == 3:
                x, target, idx = batch
            else:
                x, target = batch
                idx = torch.arange(len(target))
            x, target = x.cuda(), target.cuda()
            optimizer.zero_grad()
            _, feat, out = net(x)
            if isinstance(out, tuple):
                out = out[-1]
            if out.dim() == 1:
                out = out.unsqueeze(0)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            if feat_dim is None:
                feat_dim = feat.shape[1]
                
            ce_losses = ce_loss_fn(out, target).detach().cpu().numpy().reshape(-1, 1)
            # === Optimization: Use SCE loss directly when batch size is 1 ===
            if len(ce_losses) == 1:
                sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
                loss = sce_criterion(out, target)
                if torch.isnan(loss):
                    print('[NaN Warning] loss is NaN! Skipping backward.')
                    continue
                loss.backward()
                optimizer.step()
                continue  # Skip subsequent clean/noisy/OT/sep logic
                
            if np.isnan(ce_losses).any():
                print('[NaN Warning] ce_losses contains NaN! Skipping this batch.')
                continue
                
            gmm = GaussianMixture(n_components=2, random_state=0)
            gmm.fit(ce_losses)
            labels_gmm = gmm.predict(ce_losses)
            means = gmm.means_.flatten()
            clean_label = np.argmin(means)
            clean_mask = torch.tensor(labels_gmm == clean_label, dtype=torch.bool, device=out.device)
            noisy_mask = ~clean_mask
            
            # Symmetric CE loss for all samples
            sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
            clean_loss = sce_criterion(out, target)
            
            # === Accumulate local clean sample features ===
            for c in range(num_classes):
                mask = (target == c) & clean_mask
                if mask.sum() > 0:
                    center_sum = feat[mask].sum(dim=0).detach().cpu()
                    if local_center_sum[c] is None:
                        local_center_sum[c] = center_sum
                    else:
                        local_center_sum[c] += center_sum
                    local_center_count[c] += mask.sum().item()
            
            # === Build local centers for OT (without considering global centers) ===
            final_centers = []
            final_classes = []
            for c in range(num_classes):
                # Only use locally accumulated centers
                if local_center_count[c] > 0:
                    local_center = local_center_sum[c] / local_center_count[c]
                    local_center = local_center.to(feat.device)
                    final_centers.append(local_center)
                    final_classes.append(c)
            
            # OT feature alignment loss for noisy samples
            if noisy_mask.sum() > 0 and len(final_centers) > 0:
                feat_noisy = feat[noisy_mask]
                if torch.isnan(feat_noisy).any():
                    print('[NaN Warning] feat_noisy contains NaN!')
                    print(feat_noisy)
                final_centers_tensor = torch.stack(final_centers, dim=0)
                ot_loss_val = center_loss_cls(final_centers_tensor, feat_noisy, None, len(final_classes))
            else:
                ot_loss_val = torch.tensor(0.0, device=out.device, requires_grad=True)
            

            
            loss = clean_loss + lambda_ot * ot_loss_val 
            if torch.isnan(loss):
                print('[NaN Warning] loss is NaN! Skipping backward.')
                continue
            loss.backward()
            optimizer.step()
    
    net.to('cpu')


def center_separation_loss(class_centers):
    """Class center separation loss"""
    # Only keep non-NaN centers
    valid_mask = ~torch.isnan(class_centers).any(dim=1)
    valid_centers = class_centers[valid_mask]
    if valid_centers.size(0) < 2:
        return torch.tensor(0.0, device=class_centers.device, requires_grad=True)
    dist_matrix = torch.cdist(valid_centers, valid_centers, p=2)
    mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1)
    separation = dist_matrix * mask
    return -separation.sum() / mask.sum()


def fedrot_local_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger):
    """FedROT-Local algorithm main function, does not transmit global class centers, only computes OT loss locally"""
    best_test_acc = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_loss = float('inf')
    record_test_acc_list = []
    record_f1_list = []
    record_precision_list = []
    record_recall_list = []
    record_loss_list = []
    n_parties = args.n_parties

    # Warmup phase
    warmup_rounds = getattr(args, 'fedot_warmup', 5)
    for round in range(warmup_rounds):
        logger.info(f"FedROT-Local Warmup round {round}")
        party_list_this_round = party_list_rounds[round]
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        global_w = global_model.state_dict()
        
        for net in nets_this_round.values():
            net.load_state_dict(global_w)
        
        # Use Symmetric CE loss for warmup training
        for net_id, net in nets_this_round.items():
            train_dl_local = train_local_dls[net_id]
            sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=args.num_classes if hasattr(args, 'num_classes') else args.num_class)
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
            net.cuda()
            net.train()
            
            for epoch in range(args.epochs):
                for x, target, idx in train_dl_local:
                    x, target = x.cuda(), target.cuda()
                    optimizer.zero_grad()
                    target = target.long()
                    _, _, out = net(x)
                    if out.dim() == 1:
                        out = out.unsqueeze(0)
                    loss = sce_criterion(out, target)
                    loss.backward()
                    optimizer.step()
            net.to('cpu')
        
        # Aggregation
        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
        global_w = None
        for idx, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if idx == 0:
                global_w = {k: v * fed_avg_freqs[idx] for k, v in net_para.items()}
            else:
                for k in global_w:
                    global_w[k] += net_para[k] * fed_avg_freqs[idx]
        global_model.load_state_dict(global_w)
        global_model.cuda()
        
        # Testing
        test_acc, f1, precision, recall, conf_matrix, avg_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc)
        record_f1_list.append(f1)
        record_precision_list.append(precision)
        record_recall_list.append(recall)
        record_loss_list.append(avg_loss)
        global_model.to('cpu')
        
        # Update best results
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if f1 > best_f1:
            best_f1 = f1
        if precision > best_precision:
            best_precision = precision
        if recall > best_recall:
            best_recall = recall
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Print current round and best results
        logger.info(f'Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        logger.info(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')
        print(f'Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        print(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')
    
    print("#####  warmup_stage ends, fedrot-local stage starts #####")
    logger.info("#####  warmup_stage ends, fedrot-local stage starts #####")
    
    # FedROT-Local main process
    for round in range(warmup_rounds, n_comm_rounds):
        logger.info(f"FedROT-Local Round {round}")
        party_list_this_round = party_list_rounds[round]
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        global_w = global_model.state_dict()
        
        for net in nets_this_round.values():
            net.load_state_dict(global_w)
        
        # Client local training (does not rely on global class centers)
        for net_id, net in nets_this_round.items():
            train_dl_local = train_local_dls[net_id]
            # Use local OT training, no transmission of global class centers
            train_net_fedot_local(net, train_dl_local, args.epochs, args, device)
        
        # Aggregation (no class center transmission involved)
        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
        global_w = None
        for idx, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if idx == 0:
                global_w = {k: v * fed_avg_freqs[idx] for k, v in net_para.items()}
            else:
                for k in global_w:
                    global_w[k] += net_para[k] * fed_avg_freqs[idx]
        global_model.load_state_dict(global_w)
        global_model.cuda()
        
        # Testing
        test_acc, f1, precision, recall, conf_matrix, avg_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc)
        record_f1_list.append(f1)
        record_precision_list.append(precision)
        record_recall_list.append(recall)
        record_loss_list.append(avg_loss)
        global_model.to('cpu')
        
        # Update best results
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if f1 > best_f1:
            best_f1 = f1
        if precision > best_precision:
            best_precision = precision
        if recall > best_recall:
            best_recall = recall
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Print current round and best results
        logger.info(f'Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        logger.info(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')
        print(f'Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        print(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')
    
    # After training, calculate the average of the last 10 rounds
    def last_k_avg(lst, k=10):
        import numpy as np
        return np.mean(lst[-k:]) if len(lst) >= k else np.mean(lst)
    
    avg_acc = last_k_avg(record_test_acc_list)
    avg_f1 = last_k_avg(record_f1_list)
    avg_precision = last_k_avg(record_precision_list)
    avg_recall = last_k_avg(record_recall_list)
    avg_loss = last_k_avg(record_loss_list)
    
    print(f'Last 10 rounds average: Acc={avg_acc:.4f}, F1={avg_f1:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, Loss={avg_loss:.4f}')
    logger.info(f'Last 10 rounds average: Acc={avg_acc:.4f}, F1={avg_f1:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, Loss={avg_loss:.4f}')
    
    return record_test_acc_list, best_test_acc