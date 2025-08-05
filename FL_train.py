import os
import sys
import json
import copy
import torch
import random
import logging
from datetime import datetime
import numpy as np
from args import get_args
from disco import *
from algorithms import *
from utils.model import *
from utils.utils import *
from algorithms.fedrot_local import fedrot_local_alg


if __name__ == '__main__':
    # Configurations
    args = get_args()
    if args.dataset_form == "imagefolder":
        if args.dataset=='RS-5':
            args.datadir='./dataset/FedRS-5'
        elif args.dataset == 'RS-15':
            args.datadir = './dataset/FedRS'
    elif args.dataset_form == "parquet":
            args.datadir = './dataset/FedRS/'
    if args.alg == 'feddisco':
        args.disco = 1
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    now_time = datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    print(now_time)

    dataset_logdir = os.path.join(args.logdir, args.dataset)
    mkdirs(dataset_logdir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % (now_time)
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(dataset_logdir, argument_path), 'w') as f:
        json.dump(str(args), f)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_path = f'log_{args.alg}_{args.dataset}_{args.model}_{args.partition}_val{args.val_distribution}_noise{args.noise_rate}_{args.noise_pattern}_{now_time}.log'
    
    logging.basicConfig(
        filename=os.path.join(dataset_logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')
    logger = logging.getLogger()
    device = torch.device(args.device)
    seed = args.init_seed
    logger.info("#" * 100)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)




    # Data partition of clients
    print("Training set partition")
    net_dataidx_map, traindata_cls_counts, y_train = partition_data(args)
    print(f"############### All Y Train Shape: {len(y_train)} ###############")
    traindata_cls_counts = traindata_cls_counts.squeeze()
    
    
    
    # Label noise parameters
    noise_min = 0.0
    noise_max = args.noise_rate  # Maximum noise ratio specified by command line argument
    n_clients = args.n_parties
    if args.dataset == 'RS-5':
        n_classes = 5
        args.num_classes = 5
    elif args.dataset == 'RS-15':
        n_classes = 15
        args.num_classes = 15

    # Statistics and comparison of class distribution for each client before and after noise injection
    print("##########⚠️ Calculate the samples of all classes (before injection | after injection) ⚠️##########")
    before_counts = []
    noisy_label_distribution = {}
    for client_idx in range(n_clients):
        labels = y_train[client_idx]
        count_arr = np.bincount(labels, minlength=n_classes)
        before_counts.append(count_arr)

    # Inject label noise linearly increasing for each client
    for client_idx in range(n_clients):
        noise_rate = noise_min + (noise_max - noise_min) * client_idx / (n_clients - 1)
        y_train[client_idx] = flip_labels(
            y_train[client_idx],
            noise_rate,
            args.noise_pattern,
            n_classes,
            seed=args.init_seed
        )

    for client_idx in range(n_clients):
        after_arr = np.bincount(y_train[client_idx], minlength=n_classes)
        print(f"Client {client_idx}: {before_counts[client_idx]} | {after_arr}")
        noisy_label_distribution[client_idx] = after_arr.tolist()

    # Add the class distribution after noise injection to args
    args.noisy_label_distribution = noisy_label_distribution
        
    
    
    # Choose clients per round
    n_party_per_round = args.sample_num
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)






    # Get the testing dataloader
    test_dl = get_test_dataloader(args)

    global_dist = np.ones(traindata_cls_counts.shape[1])/traindata_cls_counts.shape[1]
    print("Length of testing set:", len(test_dl.dataset))







    # Local models initialization
    nets = init_nets(args.n_parties, args, device='cpu')
    global_models = init_nets(1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file:
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round





    # Get the training dataloaders
    print("####################################⚠️ Warning: Use the Injected Noisy Labels for Training ⚠️####################################")
    train_local_dls=[]
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        train_dl_local= get_train_dataloader(args, net_id, custom_labels=y_train[net_id])
        train_local_dls.append(train_dl_local)
        all_labels = []
        
    acc_dir = os.path.join(dataset_logdir, 'acc_list')
    if not os.path.exists(acc_dir):
        os.mkdir(acc_dir)
    acc_path = os.path.join(dataset_logdir, f'acc_list/{args.alg}_{args.dataset}_{args.model}_{args.partition}_val{args.val_distribution}_{now_time}.npy')
    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    else:
        moment_v = None
        
        
        
        
        
        
        
    # Training
    print("\nTraining begins!")
    if args.alg == 'moon':
        print("----MOON----\n")
        record_test_acc_list, best_test_acc = moon_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger)
    elif args.alg == 'fedavg' or args.alg == 'fedavg-dp':
        print("----FEDAVG----\n")
        record_test_acc_list , best_test_acc = fedavg_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger)   
    elif args.alg == 'scaffold':
        print("----SCAFFOLD----\n")
        record_test_acc_list, best_test_acc = scaffold_alg(args, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger)
    elif args.alg == 'feddyn':
        print("----FEDDYN----\n")
        record_test_acc_list, best_test_acc = feddyn_alg(args, n_comm_rounds, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger)
    elif args.alg == 'feddc':
        print("----FEDDC----")
        record_test_acc_list, best_test_acc = feddc_alg(args, n_comm_rounds, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger)
    elif args.alg == 'fednova':
        print("---FEDNOVA---")
        record_test_acc_list, best_test_acc = fednova_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger)
    elif args.alg == 'fedprox':
        print("---FEDPROX---\n")
        record_test_acc_list, best_test_acc = fedprox_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger)
        
    
    
    elif args.alg == 'feddisco':
        print("---FEDDISCO---\n")
        args.disco = 1
        record_test_acc_list , best_test_acc = fedavg_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger)   
        
    elif args.alg == 'fedavgm':
        print("---FEDAVGM---\n")
        record_test_acc_list , best_test_acc = fedavg_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger)   



    #TODO: Robust Aggregation Methods
    elif args.alg == 'symmetricCE':
        print("---Symmetric CE---\n")
        record_test_acc_list , best_test_acc = fedavg_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger)   


    elif args.alg == 'fedrot-local':
        print("--- [ours] FEDROT-Local---\n")
        record_test_acc_list, best_test_acc = fedrot_local_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger)
        
    
    np.save(acc_path, np.array(record_test_acc_list))
    print('>> Global Model Best accuracy: %f' % best_test_acc)
    print(args)
    print(now_time)
