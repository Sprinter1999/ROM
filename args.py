import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # initialization
    parser.add_argument('--init_seed', type=int, default=0, help="random seed")
    parser.add_argument('--device', type=str, default='cuda:0', help='the device to run the program')

    # log
    parser.add_argument('--log_file_name', type=str, default=None, help='the log file name')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./checkpoints/", help='model directory path')

    # benchmark
    parser.add_argument('--download_data', type=int, default=0, help='whether to download the dataset')
    parser.add_argument('--dataset', type=str, default='RS-15', help='dataset used for training')
    parser.add_argument('--dataset_form', type=str, default='imagefolder', help='the storage method for the dataset')
    parser.add_argument('--val_distribution', type=int, default='1', help='which distribution to use for validating')
    parser.add_argument('--datadir', type=str, required=False, default="../dataset/", help="data directory")
    parser.add_argument('--n_parties', type=int, default=135, help='number of workers in a distributed cluster')    
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication round') 
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')   
    parser.add_argument('--partition', type=str, default='noniid-1', choices=['noniid-1', 'noniid-2'], help='the data partitioning strategy')
    parser.add_argument('--beta', type=float, default=0.5, help='the parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--n_niid_parties', type=int, default=5, help='number of niid workers')     
    parser.add_argument('--train_global_imb', type=int, default=0, help='the imbalance ratio of global training set, 0 denotes uniform')
    
    # general parameters in training
    parser.add_argument('--alg', type=str, default='fedavg', help='federated algorithm')
    parser.add_argument('--model', type=str, default='resnet18', help='neural network used in training')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument('--save_model',type=int,default=0)
    
    
    
    #TODO: Inject label noise
    parser.add_argument('--noise_rate', type=float, default=0.4, help='label noise rate, common strategy: 0.4, 0.8')
    parser.add_argument('--noise_pattern', type=str, default='symmetric', choices=['symmetric', 'pairflip'], help='label noise pattern')

    
    # parameters of other algorithms
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--mu', type=float, default=0.01, help='the mu parameter for fedprox or moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_num', type=int, default=10,help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--dp_sigma', type=float, default=None, help='the dp parameter')
    parser.add_argument('--dp_max_grad_norm', type=float, default=1.0, help='the dp parameter')
    parser.add_argument('--dp_delta', type=float, default=1e-4, help='the dp parameter')
    parser.add_argument('--dp_epsilon', type=float, default=1.0, help='the dp parameter')
    # disco parameters
    parser.add_argument('--disco', type=int, default=0, help='whether to use disco aggregation')
    parser.add_argument('--measure_difference', type=str, default='kl', help='how to measure difference. e.g. only_iid, cosine')
    parser.add_argument('--disco_a', type=float, default=0.5, help='under sub mode, n_k-disco_a*d_k+disco_b')
    parser.add_argument('--disco_b', type=float, default=0.1)


    parser.add_argument('--gamma', type=float, default=0.75)
    parser.add_argument('--eps', type=float, default=1e-6)
    
    #For TrimmedMean
    parser.add_argument('--compromised_rate', type=float, default=0.2)
    
    #For SymmetricCE
    parser.add_argument('--sce_alpha', type=float, default=0.1, help='Symmetric CE中CE部分的权重')
    parser.add_argument('--sce_beta', type=float, default=1.0, help='Symmetric CE中RCE部分的权重')
    
    #For GCE
    parser.add_argument('--gce_q', type=float, default=0.7, help='GCE loss中的q参数')
    
    #For FedLSR
    parser.add_argument('--fedlsr_entropy', type=float, default=0.3, help='FedLSR entropy weight')
    parser.add_argument('--fedlsr_consistency', type=float, default=0.4, help='FedLSR consistency weight')
    
    #for FedNoro
    parser.add_argument('--warm_up_epochs', type=int, default=10, help='FedNoRo warm-up epochs')
    
    
    #for fedROT
    parser.add_argument('--ot_loss_weight', type=float, default=0.5, help='ot loss weight')
    parser.add_argument('--fedot_warmup', type=int, default=20, help='ot threshold for confidence') #
    parser.add_argument('--center_sep_weight', type=float, default=0.05, help='weight for center separation loss in fedrot')
    
    # Visualization
    parser.add_argument('--tsne', action='store_true', help='generate t-SNE visualization after training')
    
    
    args = parser.parse_args()
    return args