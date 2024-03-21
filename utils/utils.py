import datetime, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import wntr
import torch
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
from python_calamine import get_sheet_data
from torch_scatter import scatter 


class WaterFutureScenario:
    """
    Object that loads the Waterfutures Scenario.
    ...    

    Methods
    -------
    
    get_demands():
        Return demands of the scenario.    
    get_heads():
        Return heads of the scenario.
    
    """

    def __init__(self, directory):
        
        """
        directory : str - directory of the measurements.            
        """
        
        self.__directory = directory
        self.__time_start = None
        self.__time_step_seconds = None
        self.__demands = None
        self.__flows = None
        self.__heads = None

    def __load_time_info(self):
        tmp_df = self.get_heads()
        self.__time_start = tmp_df.index[0]
        time_step = tmp_df.index[1]
        self.__time_step_seconds = (time_step - self.__time_start).total_seconds()

    def __timestamp_to_idx(self, timestamp):
        if self.__time_start is None or self.__time_step_seconds is None:
            self.__load_time_info()
        
        date_format = '%Y-%m-%d %H:%M:%S'
        cur_time = datetime.datetime.strptime(timestamp, date_format)
        return int((cur_time - self.__time_start).total_seconds() / self.__time_step_seconds)

    def __load_flows(self):
        ''' Load the flows for this scenario and save in internal variable. '''
        path_to_flows = self.__directory

        try:
            tmp_df = pd.read_csv(path_to_flows)
        except:
            recs: list[list] = get_sheet_data(path_to_flows, sheet=2)
            tmp_df = pd.DataFrame.from_records(recs, coerce_float=True)
            tmp_df.columns = tmp_df.iloc[0]
            tmp_df = tmp_df.drop(tmp_df.index[0])
        
        tmp_df['Timestamp'] = pd.to_datetime(tmp_df['Timestamp'])
        tmp_df = tmp_df.set_index("Timestamp")
        self.__flows = tmp_df.astype("float32") 
       
    def get_flows(self):
        if self.__flows is None:
            self.__load_flows()
        return self.__flows.copy() 

    def __load_demands(self):
        ''' Load the demands for this scenario and save in internal variable. '''
        path_to_demands = self.__directory

        try:
            tmp_df = pd.read_csv(path_to_demands)
        except:
            recs: list[list] = get_sheet_data(path_to_demands, sheet=5)
            tmp_df = pd.DataFrame.from_records(recs, coerce_float=True)
            tmp_df.columns = tmp_df.iloc[0]
            tmp_df = tmp_df.drop(tmp_df.index[0])
       
        tmp_df['Timestamp'] = pd.to_datetime(tmp_df['Timestamp'])
        tmp_df = tmp_df.set_index("Timestamp")
        self.__demands = tmp_df.astype("float32")
       
    def get_demands(self):
        if self.__demands is None:
            self.__load_demands()
        return self.__demands.copy()   

    def __load_heads(self):
        path_to_heads = self.__directory
        try:
            tmp_df = pd.read_csv(path_to_heads)
        except:
            recs: list[list] = get_sheet_data(path_to_heads, sheet=4)
            tmp_df = pd.DataFrame.from_records(recs, coerce_float=True)
            tmp_df.columns = tmp_df.iloc[0]
            tmp_df = tmp_df.drop(tmp_df.index[0])

        tmp_df['Timestamp'] = pd.to_datetime(tmp_df['Timestamp'])
        tmp_df = tmp_df.set_index("Timestamp")
        self.__heads = tmp_df.astype("float32")
    
    def get_heads(self):
        if self.__heads is None:
            self.__load_heads()
        return self.__heads.copy()
    


def convert_to_bi_edges(edge_index, edge_attr=None):    
    """ 
        Method to convert directed edges to bi-directional edges.
    """    
    edge_index_swap = edge_index.clone()
    edge_index_swap_copy = edge_index_swap.clone()
    edge_index_swap[0,:] = edge_index_swap_copy[1,:]
    edge_index_swap[1,:] = edge_index_swap_copy[0,:]   
    edge_index_bi = torch.cat([edge_index, edge_index_swap], dim=-1)
    if edge_attr is not None:
        edge_attr_bi = torch.cat([edge_attr, edge_attr], dim=1)
        return edge_index_bi, edge_attr_bi
    else:
        return edge_index_bi
    

def relabel_nodes(node_indices, edge_indices):
    """ Relabels node and edge indices starting from 0. """

    node_idx = np.where(node_indices >= 0)[0]
    edge_idx = copy.deepcopy(edge_indices)
    for idx, index in zip(node_idx, node_indices):
        edge_idx = np.where(edge_indices == index, idx, edge_idx)
    edge_idx = torch.tensor(edge_idx, dtype=int)

    return node_idx, edge_idx



def compute_net_flows(h, r, edge_index, zeta=1e-32):
    """ 
        Computing flows and demands from heads using 
        hydraulic principles (eq. 8). 
    """
        
    sndr_node_attr = h[edge_index[0, :], :] 
    rcvr_node_attr = h[edge_index[1, :], :] 
    h_l = sndr_node_attr - rcvr_node_attr 
    h_lr = h_l / r
    h_lr = torch.nan_to_num(h_lr, nan=0, posinf=0, neginf=0)
    q = (torch.pow(h_lr.abs() + zeta, 1/1.852) * torch.sign(h_lr)) + zeta

    d = scatter(q, dim=0, index=edge_index[1:2, :].T, reduce='add', out=torch.zeros_like(h))  

    return d, q       


def construct_heads(J, h, q, r, edge_index, zeta=1e-32):   
    """ 
        Reconstructing heads using our physics-informed algorithm (eq. 6 & 7). 
    """            
            
    q_x = (torch.pow(q.abs() + zeta, 1.852) * torch.sign(q)) + zeta
    h_l = q_x * r
    for _ in range(J):
        sndr_node_attr = h[edge_index[0, :], :]
        h = scatter(sndr_node_attr - torch.relu(h_l), dim=0, index=edge_index[1:2, :].T, reduce='max', out=h.clone())

    return h


def create_graph(inp_file, path_to_data):
    """ Reads the WDS and scenarios as graphs.

        It requires a path to the Network Structure file *.inp (inp_file) and 
        a path to the saved simulations *.xlsx file (path_to_data).

        It returns a "wdn_graph" object consisting of node features (X), node coordinates,
        node indices, edge indices and edge attributes.    
     """
    """ Reading the .inp file using wntr package. """
    wn = wntr.network.WaterNetworkModel(inp_file)

    """ Reading edge attributes."""
    n_edges = len(wn.get_graph().edges())
    edge_names = np.array(wn.link_name_list)
    edges_df = pd.DataFrame(index=edge_names, dtype=float)
    edges_df = pd.concat((edges_df, 
                          wn.query_link_attribute('length'),
                          wn.query_link_attribute('diameter'),
                          wn.query_link_attribute('roughness')), 
                          axis=1).reset_index()
    edges_df.columns = ['names', 'length', 'diameter', 'roughness']
    all_edge_indices = np.zeros((2, n_edges), dtype=int)
    
    """ Reading node indices. """
    nodes_df = pd.DataFrame(index=wn.get_graph().nodes())
    node_indices = np.arange(nodes_df.shape[0])
    n_nodes = len(node_indices)

    """ Saving reservoir indices. """
    reservoirs = []
    for node_id, idx in zip(wn.node_name_list, node_indices):
        if wn.nodes._data[node_id].node_type == 'Reservoir':  
            reservoirs.append(idx)   

    """ Reading node attributes."""
    nodes_df = pd.concat((nodes_df, wn.query_node_attribute('elevation'), 
                          wn.query_node_attribute('coordinates')), axis=1).reset_index()
    nodes_df.columns = ['names', 'elevation', 'coordinates']
    node_coords = nodes_df['coordinates'].values

    """ Reading edge indices. """
    for idx, (name, link) in enumerate(wn.links()):
        all_edge_indices[0, idx] = nodes_df.loc[nodes_df['names'] == link.start_node_name].index.values
        all_edge_indices[1, idx] = nodes_df.loc[nodes_df['names'] == link.end_node_name].index.values

    """ Reading the scenario. """
    scenario = WaterFutureScenario(path_to_data)       

    """ 
    Creating a (S x N_n x 2) tensor having EPANET/WNTR estimated Heads (h_wntr) 
    and the Original Demands (d_star).
    """
    X_df_heads = scenario.get_heads()
    X = torch.zeros(X_df_heads.shape[0], n_nodes, 2, dtype=torch.float32)
    X[:, :, 0] = torch.tensor(X_df_heads.values, dtype=torch.float32)
    X_df_demands = scenario.get_demands()
    X[:, :, 1] = torch.tensor(X_df_demands.values, dtype=torch.float32)  
    X = X.nan_to_num(0)
    
    """ Relabelling edge indices for consistency. """
    node_indices_relabeled, edge_indices_relabeled = relabel_nodes(node_indices, all_edge_indices)    
   
    """ Computing edge attribute r. """
    edge_attr_directed = 10.667 * \
                     torch.tensor(edges_df.length.values) * \
                     torch.pow(torch.tensor(edges_df.roughness.values), -1.852) * \
                     torch.pow(torch.tensor(edges_df.diameter.values), -4.871)
    edge_attr_directed = torch.nan_to_num(edge_attr_directed, nan=0, posinf=0, neginf=0).repeat(X.shape[0], 1) 
    edge_attr_directed = torch.tensor(edge_attr_directed, dtype=torch.float32).unsqueeze(2)

    """ Converting directed edge indices and attributes to bidirectional. """
    edge_indices, edge_attr = convert_to_bi_edges(edge_indices_relabeled, edge_attr_directed)                
    edge_indices = edge_indices.repeat(X.shape[0], 1, 1)

    """ 
    Reading flows estimated by EPANET/WNTR and appending those to edge attributes
    to create a (S x N_e x 2) tensor. 
    """
    flows_df = scenario.get_flows() 
    flows = torch.tensor(flows_df.values, dtype=torch.float32).unsqueeze(2)
    flows = torch.cat((flows, flows * -1), dim=1)
    edge_attr = torch.cat((edge_attr, flows), dim=-1)
    
    """ Creating the graph object and returning it along with the indices of the reservoirs. """    
    wdn_graph = WDN_Graph(X, node_coords, node_indices, edge_indices, edge_attr)

    return wdn_graph, reservoirs



class WDN_Graph():
    def __init__(self, X=None, node_coords=None, node_indices=None, edge_indices=None, edge_attr=None,):
        super().__init__()   
        """
        A graph object with following attributes:
            X               A (S x N_n x 2) tensor having EPANET/WNTR estimated Heads (h_wntr) and the Original Demands (d_star).
            node_coords     Coordinates of nodes that are useful for plotting,
            node_indices    Indices of the nodes.     
            edge_indices    A (S x 2 x N_e) tensor specifying bidirectional edge connections.  
            edge_attr       A (S x N_e x 2) tensor having r and the flows estimated by EPANET/WNTR (q_wntr)
        """
        self.X = X
        self.node_coords = node_coords
        self.node_indices = node_indices     
        self.edge_indices = edge_indices
        self.edge_attr = edge_attr



def load_dataset(wds, n_nodes, reservoirs):
    """ 
        Creating and loading the Pytorch in-memory dataset.
    """
    dataset = WDN_Dataset_IM()
    dataset._data_list = np.arange(wds.X.shape[0])
    Y = dataset.load(wds, reservoirs, n_nodes)    
    return dataset, Y  


class WDN_Dataset_IM(InMemoryDataset):
    """ 
        InMemory Dataset Object.

        Creates a list of separate graphs for each sample. 
        Each graph is characterized by:
            x           a (N_n x 2) tensor with masked h_wntr and d_star.
            y           a (N_n x 2) tensor with h_wntr and d_star.
            edge_index  a (2 x N_e) edge indices tensor.      
            edge_attr   a (N_e x 2) edge attributes tensor with r and q_wntr. 
    """ 
    def __init__(self, ):
        super().__init__()        

    def len(self):
        return len(self._data_list)
    
    def get(self, idx):
        wdn_graph = self.data[idx]
        return wdn_graph

    def load(self, wds, reservoirs, n_nodes):
        
        self.data = []
        Y = wds.X.clone()
        
        for idx in self._data_list:   
            mask = torch.zeros((n_nodes, 1), dtype=torch.float32) 
            mask[reservoirs] = 1 
            wdn_graph = Data(
                            x = wds.X[idx, :, :].clone(),
                            y = Y[idx, :, :].clone(),  
                            edge_attr = wds.edge_attr[idx],
                            edge_index = wds.edge_indices[idx]
                            )    
            wdn_graph.x[mask[:,0] == 0, 0] = 0 
            self.data.append(wdn_graph)
        return Y
    
def mean_rel_abs_error(y, y_hat):
    """ 
        Computing the Mean Relative Absolute Error as defined in the paper.
        Here the mean across nodes is returned.
    """
    mrae = (y - y_hat).abs() / y.abs()
    mrae = torch.nan_to_num(mrae, nan=0, posinf=0, neginf=0)
    return mrae.mean(dim=1)

def get_not_oulier_errors(errors, not_outlier_parcent):
    """ 
        Computing the Mean Relative Absolute Error excluding 5% outliers 
        as defined in the paper. Here the mean across nodes is returned.
    """
    errors_sorted, _ = torch.sort(errors, dim=0)
    not_outlier_nodes = int(errors_sorted.shape[0] * not_outlier_parcent)
    errors_no = errors_sorted[: not_outlier_nodes]    
    return errors_no.mean(dim=1) 


def pearson_coef(y, y_predict):
    """
        Computes the Pearson Correlation Coefficient.
    """
    n_nodes = y.shape[1]
    p_coef = np.zeros((n_nodes))
    for node in range(n_nodes):
        y_diff = y[:, node] - y[:, node].mean(dim=0)
        y_predict_diff = y_predict[:, node] - y_predict[:, node].mean(dim=0)
        p_coef_node = (y_diff * y_predict_diff).sum(dim=0) / \
                (torch.sqrt((y_diff ** 2).sum(dim=0)) * torch.sqrt((y_predict_diff ** 2).sum(dim=0)) + 1e-32)
        p_coef[node] = p_coef_node.item()
        
    return p_coef


def plot_graph(inp_file, e_index, args, save_dir="", node_errors=[], node_labels=None, 
               plot=True, with_labels=True, cmap="summer", flag="orig", edge_errors=None,
               node_font_size=14, edge_font_size=7, arrows=True, node_size=500, 
               node_names=False, width=3, arrowsize=3, figsize=(60, 37)):
    """
        Plots the WDS with a spectrum of colors indicating the level of error for every node
    """
    wn = wntr.network.WaterNetworkModel(inp_file)
    G = nx.DiGraph()
    edge_list = [ (u, v) for u, v in zip(*np.array(e_index)) ]
    G.add_edges_from(edge_list)
    node_list = range(G.number_of_nodes())

    if plot:
        pos = wn.query_node_attribute('coordinates').values#[:-3]
        fig, ax = plt.subplots(figsize=figsize)
        node_color = node_errors
        if node_labels is None:
            if node_names:
                node_labels = wn.node_name_list
            else:
                node_labels = node_list
        if edge_errors is not None:
            edge_color = edge_errors
        else:
            edge_color = 'royalblue'
        edge_cmap = mpl.cm.get_cmap(name='Reds')

        nx.draw_networkx(G, 
            pos=pos,
            node_color=list(node_color),
            nodelist=node_list,
            labels={ n: l for n, l in zip(node_list, node_labels) },
            cmap=cmap,
            node_size=node_size, 
            ax=ax,
            edgelist=edge_list,
            width=width, 
            linewidths=width,
            edge_color=edge_color, 
            edge_vmin=0., 
            edge_vmax=1.,  
            # edge_cmap=edge_cmap,
            with_labels=with_labels,
            font_size=node_font_size,
            arrows=arrows,
            arrowsize=arrowsize
            )
        if edge_errors is not None:
            edge_labels = { (u, v) : e for u, v, e in zip(*np.array(e_index), np.array(edge_errors)) }
            nx.draw_networkx_edge_labels(G,
                pos = pos,
                edge_labels = edge_labels,
                font_size = edge_font_size,
            )

        plt.savefig(save_dir+"/_graph_"+args.model+"_"+str(args.n_aggr)+"_"+str(args.n_epochs)+"_"+str(datetime.date.today())+"_"+flag+".pdf")
        #plt.close()

def plot_timeseries(Y, Y_hat, mask, args={}, save_dir="", flag="test", scatter=False):
    plt.figure(figsize=(25,50))
    t = np.arange(Y.shape[0])    
    for node in range(Y.shape[1]):
        plt.subplot(10, 5, node+1)
        plt.plot(t, Y[:, node], label="Ground Truth", color="orange")
        if scatter:
            Y_hat[Y_hat == 0] = np.nan
            plt.scatter(t, Y_hat[:, node], label="Prediction", color="green", s=[0.75 for n in range(len(t))])
        else:
            plt.plot(t, Y_hat[:, node], label="Prediction", color="green")
        plt.xlabel("Time")
        plt.ylabel("Pressure")
        plt.title(str(mask[node]))
        plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir+"/node_timeseries_"+str(args.I)+"_"+str(args.n_epochs)+"_"+str(datetime.date.today())+"_"+flag+".jpg")
    plt.close()

