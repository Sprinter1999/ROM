import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from algorithms.symmetricCE import SCELoss
from algorithms.otloss import center_loss_cls
from utils.utils import compute_accuracy


def train_net_fedrom_dist(net, train_dataloader, epochs, args, device):
    """Local training with distance-only GMM reliability: only center-distance-GMM."""
    net.cuda()
    net.train()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    lambda_ot = getattr(args, 'ot_loss_weight', 1.0)
    num_classes = args.num_classes if hasattr(args, 'num_classes') else args.num_class

    for epoch in range(epochs):
        local_center_sum = {c: None for c in range(num_classes)}
        local_center_count = {c: 0 for c in range(num_classes)}

        for batch in train_dataloader:
            if len(batch) == 3:
                x, target, idx = batch
            else:
                x, target = batch
            x, target = x.cuda(), target.cuda()
            optimizer.zero_grad()
            _, feat, out = net(x)
            if isinstance(out, tuple):
                out = out[-1]
            if out.dim() == 1:
                out = out.unsqueeze(0)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)

            ce_losses = ce_loss_fn(out, target).detach().cpu().numpy().reshape(-1, 1)
            if len(ce_losses) == 1:
                sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
                loss = sce_criterion(out, target)
                if torch.isnan(loss):
                    continue
                loss.backward(); optimizer.step(); continue

            if np.isnan(ce_losses).any():
                continue

            # Use all samples for center accumulation (no loss-based filtering)
            for c in range(num_classes):
                mask = (target == c)
                if mask.sum() > 0:
                    center_sum = feat[mask].sum(dim=0).detach().cpu()
                    if local_center_sum[c] is None:
                        local_center_sum[c] = center_sum
                    else:
                        local_center_sum[c] += center_sum
                    local_center_count[c] += mask.sum().item()

            final_centers = []
            for c in range(num_classes):
                if local_center_count[c] > 0:
                    local_center = local_center_sum[c] / local_center_count[c]
                    final_centers.append(local_center.to(feat.device))

            # Distance-based GMM for reliability
            if len(final_centers) > 0:
                centers_tensor = torch.stack(final_centers, dim=0)
                feat_norm = torch.nn.functional.normalize(feat, dim=1)
                centers_norm = torch.nn.functional.normalize(centers_tensor, dim=1)
                dist_mat = torch.cdist(feat_norm, centers_norm, p=2)
                dist_min, _ = dist_mat.min(dim=1)
                dist_min_np = dist_min.detach().cpu().numpy().reshape(-1, 1)
                try:
                    gmm_center = GaussianMixture(n_components=2, random_state=0)
                    gmm_center.fit(dist_min_np)
                    labels_center = gmm_center.predict(dist_min_np)
                    means_center = gmm_center.means_.flatten()
                    near_label = np.argmin(means_center)
                    near_mask = torch.tensor(labels_center == near_label, dtype=torch.bool, device=feat.device)
                except Exception:
                    near_mask = torch.ones(len(target), dtype=torch.bool, device=feat.device)
                reliability_mask = near_mask
            else:
                centers_tensor = None
                reliability_mask = torch.ones(len(target), dtype=torch.bool, device=feat.device)

            noisy_mask = ~reliability_mask

            # SCE loss for all samples
            sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
            clean_loss = sce_criterion(out, target)

            # OT-like alignment on noisy samples
            if noisy_mask.sum() > 0 and centers_tensor is not None:
                feat_noisy = feat[noisy_mask]
                ot_loss_val = center_loss_cls(centers_tensor, feat_noisy, None, len(final_centers))
            else:
                ot_loss_val = torch.tensor(0.0, device=out.device, requires_grad=True)

            loss = clean_loss + lambda_ot * ot_loss_val
            if torch.isnan(loss):
                continue
            loss.backward(); optimizer.step()

    net.to('cpu')


def fedrom_dist_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger):
    """Federated routine that uses distance-only GMM reliability locally."""
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

    warmup_rounds = getattr(args, 'fedot_warmup', 5)

    # Warmup phase: SCE only + FedAvg
    for round in range(warmup_rounds):
        logger.info(f"FedROM-Dist Warmup round {round}")
        party_list_this_round = party_list_rounds[round]
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        global_w = global_model.state_dict()
        for net in nets_this_round.values():
            net.load_state_dict(global_w)
        for net_id, net in nets_this_round.items():
            train_dl_local = train_local_dls[net_id]
            sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=args.num_classes if hasattr(args, 'num_classes') else args.num_class)
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
            net.cuda(); net.train()
            for epoch in range(args.epochs):
                for x, target, idx in train_dl_local:
                    x, target = x.cuda(), target.cuda()
                    optimizer.zero_grad(); target = target.long()
                    _, _, out = net(x)
                    if out.dim() == 1: out = out.unsqueeze(0)
                    loss = sce_criterion(out, target)
                    loss.backward(); optimizer.step()
            net.to('cpu')
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
        global_model.load_state_dict(global_w); global_model.cuda()
        test_acc, f1, precision, recall, conf_matrix, avg_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc); record_f1_list.append(f1)
        record_precision_list.append(precision); record_recall_list.append(recall); record_loss_list.append(avg_loss)
        global_model.to('cpu')
        best_test_acc = max(best_test_acc, test_acc); best_f1 = max(best_f1, f1)
        best_precision = max(best_precision, precision); best_recall = max(best_recall, recall)
        best_loss = min(best_loss, avg_loss)
        logger.info(f'Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        logger.info(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}')
        print(f'Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        print(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}')

    # Main rounds with distance-only FedROM
    for round in range(warmup_rounds, n_comm_rounds):
        logger.info(f"FedROM-Dist Round {round}")
        party_list_this_round = party_list_rounds[round]
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        global_w = global_model.state_dict()
        for net in nets_this_round.values():
            net.load_state_dict(global_w)
        for net_id, net in nets_this_round.items():
            train_dl_local = train_local_dls[net_id]
            train_net_fedrom_dist(net, train_dl_local, args.epochs, args, device)
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
        global_model.load_state_dict(global_w); global_model.cuda()
        test_acc, f1, precision, recall, conf_matrix, avg_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc); record_f1_list.append(f1)
        record_precision_list.append(precision); record_recall_list.append(recall); record_loss_list.append(avg_loss)
        global_model.to('cpu')
        best_test_acc = max(best_test_acc, test_acc); best_f1 = max(best_f1, f1)
        best_precision = max(best_precision, precision); best_recall = max(best_recall, recall)
        best_loss = min(best_loss, avg_loss)
        logger.info(f'Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        logger.info(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}')
        print(f'Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        print(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}')

    def last_k_avg(lst, k=10):
        import numpy as np
        return np.mean(lst[-k:]) if len(lst) >= k else np.mean(lst)

    avg_acc = last_k_avg(record_test_acc_list)
    avg_f1 = last_k_avg(record_f1_list)
    avg_precision = last_k_avg(record_precision_list)
    avg_recall = last_k_avg(record_recall_list)
    avg_loss = last_k_avg(record_loss_list)
    logger.info(f'Last 10 rounds average: Acc={avg_acc:.4f}, F1={avg_f1:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, Loss={avg_loss:.4f}')
    print(f'Last 10 rounds average: Acc={avg_acc:.4f}, F1={avg_f1:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, Loss={avg_loss:.4f}')
    return record_test_acc_list, best_test_acc


