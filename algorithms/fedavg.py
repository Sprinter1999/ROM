import copy
import torch
from utils.model import *
from utils.utils import *
from disco import *
from algorithms.client import local_train_net


def fedavg_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger):
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

    for round in range(n_comm_rounds):
        logger.info("in comm round:" + str(round))
        print("In communication round:" + str(round))
        party_list_this_round = party_list_rounds[round]
        if args.sample_num<args.n_parties:
            print(f'Clients this round : {party_list_this_round}')

        global_w = global_model.state_dict()
        if args.server_momentum:
            old_w = copy.deepcopy(global_model.state_dict())

        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
            net.load_state_dict(global_w)
            
            
            
        #TODO: Local update
        local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls,  test_dl=test_dl, global_model=global_model,device=device, logger=logger)
        
        
        
        
        # Aggregation weight calculation
        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
        # if round==0 or args.sample_num<args.n_parties:
        #     print(f'Dataset size weight : {fed_avg_freqs}')

        if args.disco: # Discrepancy-aware collaboration
            distribution_difference = get_distribution_difference(traindata_cls_counts, participation_clients=party_list_this_round, metric=args.measure_difference, hypo_distribution=global_dist)
            print(len(fed_avg_freqs), distribution_difference.shape)
            fed_avg_freqs = disco_weight_adjusting(fed_avg_freqs, distribution_difference, args.disco_a, args.disco_b)
            if round==0 or args.sample_num<args.n_parties:
                print(f'Distribution_difference : {distribution_difference}\nDisco Aggregation Weights : {fed_avg_freqs}')



        # Model aggregation
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if args.alg == 'fedavg-dp':
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = (net_para[key]+gaussian_noise(net_para[key].shape, args, net_para[key].device)) * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += (net_para[key]+gaussian_noise(net_para[key].shape, args, net_para[key].device)) * fed_avg_freqs[net_id]
            else:
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]



        if args.server_momentum:
            delta_w = copy.deepcopy(global_w)
            for key in delta_w:
                delta_w[key] = old_w[key] - global_w[key]
                moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                global_w[key] = old_w[key] - moment_v[key]
        global_model.load_state_dict(global_w)
        global_model.cuda()

        # Test
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
        # Use logger.info instead of print
        logger.info(f'Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        logger.info(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')
        
        
        if(best_test_acc<test_acc):
            best_test_acc=test_acc
            logger.info('New Best best test acc:%f'% test_acc)
        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
        print('>> Global Model Test accuracy: %f, Best: %f' % (test_acc, best_test_acc))
        
        mkdirs(args.modeldir+'fedavg/')
        if args.save_model:   
            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
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