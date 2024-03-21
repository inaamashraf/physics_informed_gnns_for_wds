import numpy as np
import torch
from utils.utils import load_dataset 
from torch_geometric.loader import DataLoader
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

""" Training """
def train(wds_train, wds_val, model, reservoirs, args, save_dir, out_f):
    """ Initializing number of epochs and learning rate. """
    n_epochs, LR = args.n_epochs, args.lr
    """ Initiating the Optimizer and Learning rate scheduler. """
    optimizer = Adam(model.parameters(), lr=LR)    
    LR_decay_step, LR_decay_rate = args.decay_step, args.decay_rate
    opt_scheduler = lr_scheduler.MultiStepLR(
                        optimizer, 
                        range(LR_decay_step, LR_decay_step*1000, LR_decay_step), 
                        gamma=LR_decay_rate
                        )
    
    """ Creating / reading the model path if not specified. """
    if args.model_path == None:
        model_path = save_dir+"/model_"+str(args.n_epochs)+"_"+str(args.I)+".pt"

    """ Checking if training using a partially trained model. """
    if args.warm_start and args.model_path != None:                
        model_state = torch.load(model_path)
        model.load_state_dict(model_state["model"])
        optimizer.load_state_dict(model_state["optimizer"])

    n_nodes = wds_train.X.shape[1]
    n_edges = wds_train.edge_attr[0].shape[0]

    """ 
    Loading train and validation datasets in to the DataLoader. 
    Here we use the h_wntr to get h_star by masking all values other
    than the reservoirs.
    """
    train_dataset, _ = load_dataset(wds_train, n_nodes, reservoirs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset, _ = load_dataset(wds_val, n_nodes, reservoirs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
   
    """ Train-validation loop """
    for epoch in tqdm(range(n_epochs)): 

        """ Training """
        train_losses = []
        for batch in train_loader: 
            
            model.train()
            optimizer.zero_grad()
            _ = model(batch, r_iter=args.r_iter)
            train_loss = model.loss()

            train_loss.backward()
            train_losses.append(train_loss.detach().cpu().item())

            """ Clipping gradients using the mean of the true demands as the clip value. """
            clip_val = model.d_star.mean().item()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_val)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

            optimizer.step()                     
        
        opt_scheduler.step()        

        """ Validation """
        if epoch % 10 == 0:
            model.eval()
            val_losses = []            
            for batch_val in val_loader:                
                with torch.no_grad():
                    _ = model(batch_val, r_iter=args.r_iter)
                val_loss = model.loss()
                val_losses.append(val_loss)                
                
            mean_val_losses = torch.mean(torch.stack(val_losses)).detach().cpu().item()    

            """ Printing mean training and validation losses. """
            print("Epoch ", epoch, ": Train loss: ", np.round(np.mean(train_losses), 8), \
                " Val loss: ", np.round(np.mean(mean_val_losses), 8))
            print("Epoch ", epoch, ": Train loss: ", np.round(np.mean(train_losses), 8), \
                " Val loss: ", np.round(np.mean(mean_val_losses), 8), file=out_f)
            
            
        if epoch % 100 == 0:
            """ Saving the model. """
            state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }
            print('model path:', model_path)
            print('model path:', model_path, file=out_f)
            torch.save(state, model_path)
            
    return state, model_path

""" Testing """
@torch.no_grad()
def test(wds_test, model, reservoirs, args, save_dir, out_f, zeta=1e-32):

    """ Loading the trained model. """
    if args.model_path is None:
        args.model_path = save_dir+"/model_"+str(args.n_epochs)+"_"+str(args.I)+".pt"
    model_state = torch.load(args.model_path)
    model.load_state_dict(model_state["model"])

    n_nodes = wds_test.X.shape[1]
    n_edges = wds_test.edge_attr[0].shape[0]

    """ 
    Loading train and validation datasets in to the DataLoader. 
    Here we use the h_wntr to get h_star by masking all values other
    than the reservoirs.
    """
    test_dataset, h_wntr = load_dataset(wds_test, n_nodes, reservoirs)    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    """" Evaluating and saving the results for h_tilde, q_hat and d_hat. """
    test_losses = []
    h_tilde, q_hat, d_hat = [], [], []
    model.eval()
    for batch in test_loader:
        h_tilde_batch = model(batch, r_iter=args.r_iter, zeta=zeta)
        test_loss = model.loss()
        test_losses.append(test_loss.detach().cpu().item())

        h_tilde_batch = torch.hstack(h_tilde_batch.detach().cpu().split(n_nodes)).view(n_nodes, -1, h_tilde_batch.shape[1])
        h_tilde.append(h_tilde_batch)

        q_hat_batch = torch.hstack(model.q_hat.detach().cpu().split(n_edges)).view(n_edges, -1, model.q_hat.shape[1])
        q_hat.append(q_hat_batch)

        d_hat_batch = torch.hstack(model.d_hat.detach().cpu().split(n_nodes)).view(n_nodes, -1, model.d_hat.shape[1])
        d_hat.append(d_hat_batch)

    h_tilde = torch.cat(h_tilde, dim=1).transpose(1,0)
    q_hat = torch.cat(q_hat, dim=1).transpose(1,0)
    d_hat = torch.cat(d_hat, dim=1).transpose(1,0)
    
    print("Test loss: ", np.round(np.mean(test_losses), 8))
    print("Test loss: ", np.round(np.mean(test_losses), 8), file=out_f)
    
    return h_wntr, h_tilde, q_hat, d_hat, test_losses 

