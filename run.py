import os, datetime
import numpy as np
from utils.utils import create_graph, WDN_Graph, mean_rel_abs_error, pearson_coef
from models.models import *
from train_test import train, test
import argparse, json 
import warnings
warnings.filterwarnings('ignore')
CUDA_LAUNCH_BLOCKING=1

def create_cli_parser():
    # ----- ----- ----- ----- ----- -----
    # Command line arguments
    # ----- ----- ----- ----- ----- -----
    parser  = argparse.ArgumentParser()
    parser.add_argument('--wdn',
                        default = 'l_town',
                        type    = str,
                        choices = ['hanoi', 'fossolo', 'pescara', 'l_town', 'zhijiang'],
                        help    = "specify the WDS for which you want to simulate the scenarios; default is l_town ")
    parser.add_argument('--mode',
                        default = 'train_test',
                        type    = str,
                        choices = ['train_test', 'evaluate'],
                        help    = "train_test i.e. train and test a new model, or evaluate i.e. evaluate on an already trained model; default is train_test. ")
    parser.add_argument('--warm_start',
                        default = False,
                        type    = bool,
                        help    = "specify True if you want to further train a partially trained model. model_path must also be specified; default is False.")
    parser.add_argument('--model_path',
                        default = None,
                        type    = str,
                        help    = "specify model path in case of re-training or evaluation; default is None.")
    parser.add_argument('--start_scenario',
                        default = 1,
                        type    = int,
                        help    = "specify the start scenario name, must be an integer; default is 1")
    parser.add_argument('--end_scenario',
                        default = 20,
                        type    = int,
                        help    = "specify the end scenario name, must be an integer; default is 20")
    parser.add_argument('--n_days',
                        default = 2,
                        type    = int,
                        help    = "number of days of data to be used for training; default is 2 days.")
    parser.add_argument('--batch_size',
                        default = 48,
                        type    = int,
                        help    = "mini-batch size used for training; default is 48.")
    parser.add_argument('--n_epochs',
                        default = 3000,
                        type    = int,
                        help    = "number of epochs of training; default is 3000.")    
    parser.add_argument('--lr',
                        default = 1e-4,
                        type    = float,
                        help    = "learning rate; default is 1e-4.")
    parser.add_argument('--decay_step',
                        default = 300,
                        type    = int,
                        help    = "step of the lr scheduler; default is 300.")
    parser.add_argument('--decay_rate',
                        default = 0.75,
                        type    = float,
                        help    = "decay rate of the lr scheduler; default is 0.75.")
    parser.add_argument('--I',
                        default = 5,
                        type    = int,
                        help    = "number of GCN layers; default is 5.")
    parser.add_argument('--n_iter',
                        default = 10,
                        type    = int,
                        help    = "minimum number of iterations; default is 10.")
    parser.add_argument('--r_iter',
                        default = 5,
                        type    = int,
                        help    = "maximum number of additional (random) iterations; default is 5.")
    parser.add_argument('--n_mlp',
                        default = 2,
                        type    = int,
                        help    = "number of layers in the MLP; default is 2.")
    parser.add_argument('--M_l',
                        default = 128,
                        type    = int,
                        help    = "latent dimension; default is 128.")
    return parser


def run(args):

    """ Creating directories. """
    file_dir = os.path.dirname(os.path.realpath(__file__)) 
    if not os.path.isdir(os.path.join(file_dir, "tmp")):
        os.system('mkdir ' + os.path.join(file_dir, "tmp"))
    save_dir = os.path.join(file_dir, "tmp", str(datetime.date.today()))
    if not os.path.isdir(save_dir):
        os.system('mkdir ' + save_dir)

    """ Computing the number of samples based on the specified number of days. """
    sample_rate = 60 // 30                          # number of samples in an hour
    n_samples_days = sample_rate * 24               # number of samples in a day
    n_samples = int(n_samples_days * args.n_days)   # number of samples to be used

   
    """ Initializing the model and printing the number of parameters. """
    model = PI_GCN( M_n = 2,                    # number of node features (d_star, d_hat).
                    out_dim = 1,                # out dimension is 1 since only flows are directly estimated.
                    M_e = 2,                    # number of edge features (q_hat, q_tilde).
                    M_l = args.M_l,             # specified latent dimension.
                    I = args.I,                 # number of GCN layers.
                    num_layers = args.n_mlp,    # number of NN layers used in every MLP.
                    n_iter = args.n_iter,       # minimum number of iterations.
                    bias = False                # we do not use any bias.
                    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print('Total parameters: ', total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters: ', trainable_params)

    """ Creating an output file to log progress. """
    out_f = open(save_dir+"/output_"+str(n_samples)+"_"+str(datetime.date.today())+".txt", "a")

    """ Initializing dataset lists. """
    X_tvt, edge_indices_tvt, edge_attr_tvt = [], [], []

    """ Reading scenarios one by one. """
    for s in range(args.start_scenario, args.end_scenario + 1):

        """ Specifying paths for scenario, .inp file and .xlsx file. """
        scenario_path = os.path.join(os.getcwd(),"networks",  args.wdn, "s"+str(s))
        """ Since Hanoi has all configuration (*.inp) files available (Vrachimis et al. 2018), we read directly from those.
            For the other networks, we read from the .inp files that we created while generating scenarios. These files
            have a consistent naming scheme, so we can read those using the WDS name. """
        if args.wdn == "hanoi":
            args.inp_file = os.path.join(scenario_path, "Hanoi_CMH_Scenario-" + str(s) + ".inp")
        else:
            args.inp_file = os.path.join(scenario_path, args.wdn + ".inp")
        args.path_to_data = os.path.join(scenario_path, "Results-Clean", "Measurements_All.xlsx")

        """ 
        Loading the dataset from the generated scenario. This returns a "wdn_graph" object with following attributes:
        X               A (S x N_n x 2) tensor having EPANET/WNTR estimated Heads (h_wntr) and the Original Demands (d_star).
        node_coords     Coordinates of nodes that are useful for plotting,
        node_indices    Indices of the nodes.     
        edge_indices    A (S x 2 x N_e) tensor specifying bidirectional edge connections.  
        edge_attr       A (S x N_e x 2) tensor having r and the flows estimated by EPANET/WNTR (q_wntr)
        Please note that we load h_wntr and q_wntr primarily for error computations after evaluation and also to get the
        reservoir heads (h_star). These are never used in the training of our model. 
        """
        wdn_graph, reservoirs = create_graph(args.inp_file, args.path_to_data)

        """ Saving only the number of samples specified. """
        X_s = wdn_graph.X[ : n_samples].clone()
        edge_indices_s = wdn_graph.edge_indices[ : n_samples].clone()
        edge_attr_s = wdn_graph.edge_attr[ : n_samples].clone()

        """ Appending these to dataset lists. """
        X_tvt.append(X_s)
        edge_indices_tvt += list(edge_indices_s)
        edge_attr_tvt += list(edge_attr_s)

        print('\nRead Scenario ', str(s))
    
    X_tvt = torch.vstack(X_tvt)    

    """ Creating train-val-test splits. """
    n_scenarios = args.end_scenario + 1 - args.start_scenario
    t_samples = n_scenarios * n_samples
    train_s, val_s = int(0.6 * t_samples), int(0.8 * t_samples)
    wds_tvt = WDN_Graph(X=X_tvt, edge_indices=edge_indices_tvt, edge_attr=edge_attr_tvt)
    wds_train = WDN_Graph(X=X_tvt[: train_s], edge_indices=edge_indices_tvt[: train_s], edge_attr=edge_attr_tvt[: train_s])
    wds_val = WDN_Graph(X=X_tvt[train_s : val_s], edge_indices=edge_indices_tvt[train_s : val_s], edge_attr=edge_attr_tvt[train_s : val_s])
    wds_test = WDN_Graph(X=X_tvt[val_s :], edge_indices=edge_indices_tvt[val_s :], edge_attr=edge_attr_tvt[val_s :])
    print(wds_train.X.shape, wds_val.X.shape, wds_test.X.shape)    
    
    print(device)

    """ Saving all the parameters to a file. """
    args_fname = save_dir+"/args_"+str(args.n_epochs)+"_"+str(n_samples)+"_"+str(datetime.date.today())+".json"
    with open(args_fname, 'w') as args_file:
        json.dump(vars(args), args_file, indent=4)
    
    if args.mode == "train_test":
        """ Training """
        state, model_path = train(wds_train, wds_val, model, reservoirs, args, save_dir, out_f)
        """ Testing """
        h_wntr, h_tilde, q_hat, d_hat, test_losses = test(wds_test, model, reservoirs, args, save_dir, out_f, zeta=0)

    if args.mode == "evaluate":
        wds_test = wds_tvt
        h_wntr, h_tilde, q_hat, d_hat, test_losses = test(wds_test, model, reservoirs, args, save_dir, out_f, zeta=0)
            

    """ Analysis """
    
    d_hat[:, reservoirs, 0] = 0
    d_star = wds_test.X[:, :, 1:2]
    e_attr = torch.stack((wds_test.edge_attr))
    q_wntr = e_attr[:, :, 1:2]
    mrae_samples = mean_rel_abs_error(d_star[:, :, 0], d_hat[:, :, 0])
    p_coefs = pearson_coef(d_star[:, :, 0], d_hat[:, :, 0])
    print("MRAE, Std and PCC - Demands (%): ", np.round(mrae_samples.mean().item() * 100, 6), np.round(mrae_samples.std().item() * 100, 6), np.round(np.mean(p_coefs) * 100, 6))
    print("MRAE, Std and PCC - Demands (%): ", np.round(mrae_samples.mean().item() * 100, 6), np.round(mrae_samples.std().item() * 100, 6), np.round(np.mean(p_coefs) * 100, 6), file=out_f)
    mrae_samples = mean_rel_abs_error(q_wntr[:, :, 0], q_hat[:, :, 0])
    p_coefs = pearson_coef(q_wntr[:, :, 0], q_hat[:, :, 0])
    print("MRAE, Std and PCC - Flows (%): ", np.round(mrae_samples.mean().item() * 100, 6), np.round(mrae_samples.std().item() * 100, 6), np.round(np.mean(p_coefs) * 100, 6))
    print("MRAE, Std and PCC - Flows (%): ", np.round(mrae_samples.mean().item() * 100, 6), np.round(mrae_samples.std().item() * 100, 6), np.round(np.mean(p_coefs) * 100, 6), file=out_f)
    mrae_samples = mean_rel_abs_error(h_wntr[:,:,0], h_tilde[:,:,0])
    p_coefs = pearson_coef(h_wntr[:, :, 0], h_tilde[:, :, 0])
    print("MRAE, Std and PCC - Heads (%): ", np.round(mrae_samples.mean().item() * 100, 6), np.round(mrae_samples.std().item() * 100, 6), np.round(np.mean(p_coefs) * 100, 6))
    print("MRAE, Std and PCC - Heads (%): ", np.round(mrae_samples.mean().item() * 100, 6), np.round(mrae_samples.std().item(), 6), np.round(np.mean(p_coefs) * 100, 6), file=out_f)

    
    
if __name__ == '__main__':
    parser = create_cli_parser()

    args = parser.parse_args()   
    
    print(args)
    run(args)

    