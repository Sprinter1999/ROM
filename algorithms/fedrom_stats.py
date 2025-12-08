import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from algorithms.symmetricCE import SCELoss
from algorithms.otloss import center_loss_cls
from utils.utils import compute_accuracy


def train_net_fedrom_with_stats(net, train_dataloader, epochs, args, device, client_id=None):
    """Local training with dual-GMM reliability and statistics: loss-GMM ∩ center-distance-GMM."""
    net.cuda()
    net.train()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    lambda_ot = getattr(args, 'ot_loss_weight', 1.0)
    num_classes = args.num_classes if hasattr(args, 'num_classes') else args.num_class

    # 统计信息
    stats = {
        'loss_gmm_samples': 0,
        'center_gmm_samples': 0,
        'intersection_samples': 0,
        'total_samples': 0
    }

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

            gmm_loss = GaussianMixture(n_components=2, random_state=0)
            gmm_loss.fit(ce_losses)
            labels_loss = gmm_loss.predict(ce_losses)
            means_loss = gmm_loss.means_.flatten()
            clean_label = np.argmin(means_loss)
            clean_mask = torch.tensor(labels_loss == clean_label, dtype=torch.bool, device=out.device)

            sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
            clean_loss = sce_criterion(out, target)

            for c in range(num_classes):
                mask = (target == c) & clean_mask
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
                    near_mask = torch.ones_like(clean_mask, dtype=torch.bool, device=feat.device)
                reliability_mask = clean_mask & near_mask
            else:
                centers_tensor = None
                reliability_mask = clean_mask

            noisy_mask = ~reliability_mask

            # 统计聚类信息
            if client_id is not None:
                total_samples = len(target)
                clean_samples = clean_mask.sum().item()
                near_samples = near_mask.sum().item()
                reliable_samples = reliability_mask.sum().item()
                
                stats['total_samples'] += total_samples
                stats['loss_gmm_samples'] += clean_samples
                stats['center_gmm_samples'] += near_samples
                stats['intersection_samples'] += reliable_samples

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
    return stats


def fedrom_alg_with_stats(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger):
    """FedROM algorithm with clustering statistics"""
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
    
    # 聚类统计记录
    clustering_stats = {
        'rounds': [],
        'loss_gmm_ratios': [],
        'center_gmm_ratios': [],
        'intersection_ratios': [],
        'loss_gmm_samples': [],
        'center_gmm_samples': [],
        'intersection_samples': [],
        'total_samples': [],
        # 新增：按客户端记录每轮的比例与数量，便于稳定性分析
        'per_client': []  # 元素形如 {'round': r, 'clients': {cid: {...}}}
    }

    warmup_rounds = getattr(args, 'fedot_warmup', 5)

    # Warmup rounds (no statistics)
    for round in range(warmup_rounds):
        logger.info(f"FedROM Warmup round {round}")
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

    # Main training rounds with statistics
    for round in range(warmup_rounds, n_comm_rounds):
        logger.info(f"FedROM Round {round}")
        party_list_this_round = party_list_rounds[round]
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        global_w = global_model.state_dict()
        for net in nets_this_round.values():
            net.load_state_dict(global_w)
        
        # 收集所有客户端的统计信息
        round_stats = {
            'loss_gmm_samples': 0,
            'center_gmm_samples': 0,
            'intersection_samples': 0,
            'total_samples': 0
        }
        
        per_client_stats = {}
        for net_id, net in nets_this_round.items():
            train_dl_local = train_local_dls[net_id]
            client_stats = train_net_fedrom_with_stats(net, train_dl_local, args.epochs, args, device, client_id=net_id)
            
            # 累加统计信息
            for key in round_stats:
                round_stats[key] += client_stats[key]
            
            # 记录该客户端的比例，便于观察筛选稳定性
            total = client_stats['total_samples']
            if total > 0:
                per_client_stats[net_id] = {
                    'loss_gmm_samples': client_stats['loss_gmm_samples'],
                    'center_gmm_samples': client_stats['center_gmm_samples'],
                    'intersection_samples': client_stats['intersection_samples'],
                    'total_samples': total,
                    'loss_gmm_ratio': client_stats['loss_gmm_samples'] / total,
                    'center_gmm_ratio': client_stats['center_gmm_samples'] / total,
                    'intersection_ratio': client_stats['intersection_samples'] / total,
                }
            else:
                per_client_stats[net_id] = {
                    'loss_gmm_samples': 0,
                    'center_gmm_samples': 0,
                    'intersection_samples': 0,
                    'total_samples': 0,
                    'loss_gmm_ratio': 0.0,
                    'center_gmm_ratio': 0.0,
                    'intersection_ratio': 0.0,
                }
        
        # 计算平均比例
        if round_stats['total_samples'] > 0:
            loss_gmm_ratio = round_stats['loss_gmm_samples'] / round_stats['total_samples']
            center_gmm_ratio = round_stats['center_gmm_samples'] / round_stats['total_samples']
            intersection_ratio = round_stats['intersection_samples'] / round_stats['total_samples']
            
            # 记录统计数据
            clustering_stats['rounds'].append(round)
            clustering_stats['loss_gmm_ratios'].append(loss_gmm_ratio)
            clustering_stats['center_gmm_ratios'].append(center_gmm_ratio)
            clustering_stats['intersection_ratios'].append(intersection_ratio)
            clustering_stats['loss_gmm_samples'].append(round_stats['loss_gmm_samples'])
            clustering_stats['center_gmm_samples'].append(round_stats['center_gmm_samples'])
            clustering_stats['intersection_samples'].append(round_stats['intersection_samples'])
            clustering_stats['total_samples'].append(round_stats['total_samples'])
            clustering_stats['per_client'].append({'round': round, 'clients': per_client_stats})
            
            # 输出聚类统计信息
            stats_msg = (f"FedROM Round {round} 聚类统计:\n"
                        f"  Loss-GMM 选取比例: {loss_gmm_ratio:.4f} ({round_stats['loss_gmm_samples']}/{round_stats['total_samples']})\n"
                        f"  Center-GMM 选取比例: {center_gmm_ratio:.4f} ({round_stats['center_gmm_samples']}/{round_stats['total_samples']})\n"
                        f"  交集选取比例: {intersection_ratio:.4f} ({round_stats['intersection_samples']}/{round_stats['total_samples']})")
            
            logger.info(stats_msg)
            print(stats_msg)
            
            # 输出本轮各客户端的比例，便于观察稳定性
            per_client_lines = [f"  Client {cid}: loss={vals['loss_gmm_ratio']:.4f} ({vals['loss_gmm_samples']}/{vals['total_samples']}), "
                                f"center={vals['center_gmm_ratio']:.4f} ({vals['center_gmm_samples']}/{vals['total_samples']}), "
                                f"inter={vals['intersection_ratio']:.4f} ({vals['intersection_samples']}/{vals['total_samples']})"
                                for cid, vals in per_client_stats.items()]
            per_client_msg = "FedROM Round {} 客户端筛选比例:\n".format(round) + "\n".join(per_client_lines)
            logger.info(per_client_msg)
            print(per_client_msg)
        
        # 聚合模型
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
    
    # 输出完整的聚类统计数据
    if clustering_stats['rounds']:
        logger.info("=" * 80)
        logger.info("FedROM 聚类统计完整数据 (用于可视化分析)")
        logger.info("=" * 80)
        
        # 输出轮次信息
        rounds_str = "轮次: " + ", ".join([str(r) for r in clustering_stats['rounds']])
        logger.info(rounds_str)
        print(rounds_str)
        
        # 输出 Loss-GMM 比例数据
        loss_ratios_str = "Loss-GMM 比例: " + ", ".join([f"{r:.4f}" for r in clustering_stats['loss_gmm_ratios']])
        logger.info(loss_ratios_str)
        print(loss_ratios_str)
        
        # 输出 Center-GMM 比例数据
        center_ratios_str = "Center-GMM 比例: " + ", ".join([f"{r:.4f}" for r in clustering_stats['center_gmm_ratios']])
        logger.info(center_ratios_str)
        print(center_ratios_str)
        
        # 输出交集比例数据
        intersection_ratios_str = "交集比例: " + ", ".join([f"{r:.4f}" for r in clustering_stats['intersection_ratios']])
        logger.info(intersection_ratios_str)
        print(intersection_ratios_str)
        
        # 输出样本数量数据
        total_samples_str = "总样本数: " + ", ".join([str(s) for s in clustering_stats['total_samples']])
        logger.info(total_samples_str)
        print(total_samples_str)
        
        # 计算平均比例
        avg_loss_ratio = np.mean(clustering_stats['loss_gmm_ratios'])
        avg_center_ratio = np.mean(clustering_stats['center_gmm_ratios'])
        avg_intersection_ratio = np.mean(clustering_stats['intersection_ratios'])
        
        summary_msg = (f"聚类统计总结:\n"
                     f"  Loss-GMM 平均比例: {avg_loss_ratio:.4f}\n"
                     f"  Center-GMM 平均比例: {avg_center_ratio:.4f}\n"
                     f"  交集平均比例: {avg_intersection_ratio:.4f}")
        
        logger.info(summary_msg)
        print(summary_msg)

        # 追加：输出各客户端在全程的累计比例，便于观察稳定性
        per_client_summary = {}
        for entry in clustering_stats['per_client']:
            for cid, vals in entry['clients'].items():
                if cid not in per_client_summary:
                    per_client_summary[cid] = {
                        'loss_gmm_samples': 0,
                        'center_gmm_samples': 0,
                        'intersection_samples': 0,
                        'total_samples': 0
                    }
                per_client_summary[cid]['loss_gmm_samples'] += vals['loss_gmm_samples']
                per_client_summary[cid]['center_gmm_samples'] += vals['center_gmm_samples']
                per_client_summary[cid]['intersection_samples'] += vals['intersection_samples']
                per_client_summary[cid]['total_samples'] += vals['total_samples']

        # 生成输出行
        per_client_lines = []
        for cid in sorted(per_client_summary.keys()):
            s = per_client_summary[cid]
            total = s['total_samples']
            if total > 0:
                loss_ratio = s['loss_gmm_samples'] / total
                center_ratio = s['center_gmm_samples'] / total
                inter_ratio = s['intersection_samples'] / total
            else:
                loss_ratio = center_ratio = inter_ratio = 0.0
            per_client_lines.append(
                f"  Client {cid}: loss={loss_ratio:.4f} ({s['loss_gmm_samples']}/{total}), "
                f"center={center_ratio:.4f} ({s['center_gmm_samples']}/{total}), "
                f"inter={inter_ratio:.4f} ({s['intersection_samples']}/{total})"
            )

        per_client_summary_msg = "各客户端累计筛选比例:\n" + "\n".join(per_client_lines)
        logger.info(per_client_summary_msg)
        print(per_client_summary_msg)
        
        logger.info("=" * 80)
        print("=" * 80)
    
    return record_test_acc_list, best_test_acc
