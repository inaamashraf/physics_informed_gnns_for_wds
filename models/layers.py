from typing import List
import torch
from torch import Tensor
from torch.nn import Dropout, Sequential
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

""" 
    Using partial code from torch_geometric.nn.conv.GENconv
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gen_conv.html#GENConv
"""

class MLP(Sequential):
    """
        Defining an MLP object with multiple linear layers,
        activation functions and dropout.
    """
    def __init__(self, dims: List[int], bias: bool = True, dropout: float = 0., activ=None):
        m = []
        for i in range(1, len(dims)):
            m.append(Linear(dims[i - 1], dims[i], bias=bias))

            if i < len(dims) - 1:    
                if activ is not None:
                    m.append(activ)
                m.append(Dropout(dropout))

        super().__init__(*m)



class GCN_Layer(MessagePassing):
    """
        Graph Convolutional Neural Network layer using message passing.
        The layer applies both node and edge feature update. 
    """
    def __init__(self, in_dim, out_dim, edge_dim, aggr='max', num_layers=2, bias=False, **kwargs):

        kwargs.setdefault('aggr', None)
        super().__init__(**kwargs)

        self.in_dim = in_dim        
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.aggr = aggr

        """ Defining dimensions for layers in MLP gamma. """
        edge_dims = [2*in_dim + edge_dim]
        for _ in range(num_layers - 1):
            edge_dims.append(edge_dim)

        """ Defining dimensions for layers in MLP eta. """
        node_dims = [edge_dim]
        for _ in range(num_layers - 1):
            node_dims.append(in_dim)

        """ 
            MLPs gamma and eta used for node and edge feature 
            updates respectively.
        """
        self.mlp_edges = MLP(edge_dims, bias=bias)
        self.mlp_nodes = MLP(node_dims, bias=bias)

    def forward(self, g, edge_index, z) -> Tensor:
        """ 
            Creating edge messages that are also the updated edge features 
            using node and edge features and the MLP gamma.
        """
        sndr_node_attr = g[edge_index[0, :], :]
        rcvr_node_attr = g[edge_index[1, :], :]
        m_e = self.mlp_edges(torch.selu(torch.cat((sndr_node_attr, rcvr_node_attr, z), dim=-1)))

        """ Aggregating edge messages using max aggregation. """
        m_e_aggr = scatter(m_e, dim=0, index=edge_index[1:2, :].T, reduce='max', out=torch.zeros_like(g))

        """ Updating node features using the MLP eta. """
        g = self.mlp_nodes(m_e_aggr)

        return g, m_e


