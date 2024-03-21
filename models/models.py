import torch
from torch_scatter import scatter
from utils.utils import compute_net_flows, construct_heads
from models.layers import *
from torch.nn import Linear, ModuleList, L1Loss, Module
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class PI_GCN(Module):
    """
        Our model using local GCN layers learning from a global Physics-Informed Algorithm.
    """
    def __init__(self, M_n, out_dim, M_e, M_l=128, aggr='max', \
                    I=5, bias=False, num_layers=2, n_iter=10):
        super(PI_GCN, self).__init__()

        """ Number of GCN layers. """
        self.I = I   
        """ Minimum number of iterations. """           
        self.n_iter = n_iter    

        """ 
            Linear layers alpha and beta used to embed node 
            and edge features respectively.
        """
        self.node_in = Linear(M_n, M_l, bias=bias)
        self.edge = Linear(M_e, M_l, bias=bias)   

        """ The MLP lambda used to estimate the flows q_hat. """
        self.flows_latent = MLP([3*M_l, M_l, M_l, out_dim], bias=bias, activ=None)   

        """ GCN layers """
        self.gcn_aggrs = ModuleList()
        for _ in range(I):
            gcn = GCN_Layer(M_l, M_l, M_l, aggr=aggr, bias=bias, num_layers=num_layers)
            self.gcn_aggrs.append(gcn)        

    def forward(self, data, r_iter=5, zeta=1e-32):

        """ Reading data. """
        data = data.to(device)
        x, edge_index, r = data.x, data.edge_index, data.edge_attr[:, 0:1]   
        self.n_nodes = int(data.num_nodes / data.num_graphs)
        self.n_edges = int(data.num_edges / data.num_graphs)        
        
        """ True demands d_star. """
        self.d_star = x[:, 1:2]             

        """ 
            True heads h_star where only reservoirs have values 
            and all other values are zeros. 
        """
        h_star = x[:, 0:1]                  
        
        """ Creating the reservoir mask to be used in the loss function. """
        self.reservoir_mask = h_star != 0

        """ Initializing h as h_star. """
        h = h_star.clone()

        """ Computing initial flows (q_hat_(0)) and demands (d_hat_(0)). """
        self.d_hat, self.q_hat = \
                compute_net_flows(
                            h, 
                            r, 
                            edge_index, 
                            zeta = zeta
                            )    
        
        """ Initializing q_tilde_(0) the same as q_hat_(0). """
        self.q_tilde = self.q_hat.clone()

        """
            Specifying the additional number of iterations.
            These are chosen randomly during training but are 
            set to maximum specified value in evaluation.
        """
        if self.training:
            K = self.n_iter + np.random.randint(0, r_iter)
        else:
            K = self.n_iter + r_iter

        """ Performing K iteration. """
        for _ in range(K):

            """____ f_1 ((D,Q), Theta)___"""

            """ 
                Embedding node and edge features using linear layers 
                alpha and beta respectively.
            """
            g = self.node_in(torch.selu(torch.cat((self.d_hat, self.d_star), dim=-1)))
            z = self.edge(torch.selu(torch.cat((self.q_tilde, self.q_hat), dim=-1)))

            """ 
                Multiple (I) GCN layers 
            """
            for gcn in self.gcn_aggrs:
                g, z = gcn(g, edge_index, z)

            """ Estimating flows q_hat using the MLP lambda. """
            sndr_g = g[edge_index[0, :], :]
            rcvr_g = g[edge_index[1, :], :]
            self.q_hat = self.q_hat + self.flows_latent(torch.selu(torch.cat((sndr_g, rcvr_g, z), dim=-1)))

            """ Adjusting the flows q_hat for directionality. """
            self.q_hat = torch.stack(self.q_hat.split(self.n_edges))
            self.q_hat_in = self.q_hat[:, : self.n_edges//2, :] 
            self.q_hat = torch.cat((self.q_hat_in, self.q_hat_in * -1), dim=1)
            self.q_hat = torch.cat((*self.q_hat,), dim=0)   

            """ Computing estimated demands d_hat using hydraulic principle (eq. 3). """
            self.d_hat = scatter(self.q_hat, dim=0, index=edge_index[1:2, :].T, reduce='add')
            

            """____ f_2 (h, q_hat)___"""

            """ Reconstructing heads using our physics-informed algorithm (eq. 6 & 7). """            
            J = self.I * self.n_iter
            h_tilde = construct_heads(
                            J, 
                            h_star.clone(), 
                            self.q_hat, 
                            r,
                            edge_index,
                            zeta = zeta
                            ) 

            """ Computing flows q_tilde and demand d_tilde using hydraulic principle (eq. 8). """
            self.d_tilde, self.q_tilde = \
                compute_net_flows(
                            h_tilde, 
                            r, 
                            edge_index, 
                            zeta = zeta
                            )    
           
        return h_tilde        

    def loss(self, rho=0.1, delta=0.1):
        """ L1 Loss """       
        l1loss = L1Loss(reduction='mean')

        """ Computing loss between true demands d_star and estimated demands by f_1 i.e. d_hat. """     
        loss_d_hat = l1loss(self.d_hat[~self.reservoir_mask], self.d_star[~self.reservoir_mask]) 

        """ Computing loss between true demands d_star and estimated demands at the end i.e. d_tilde. """     
        loss_d_tilde = l1loss(self.d_tilde[~self.reservoir_mask], self.d_star[~self.reservoir_mask]) 

        """ Computing loss between estimated flows q_hat (f_1) and q_tilde (f). """     
        loss_q = l1loss(self.q_hat, self.q_tilde) 

        """ Summing the losses. """
        _loss =  loss_d_hat + rho * loss_d_tilde + delta * loss_q

        return _loss

