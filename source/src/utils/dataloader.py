#load data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops
from torch_geometric.datasets import TUDataset
from scipy.sparse import csr_matrix
import networkx as nx
import pickle
from utils.utils import*

def label_process(labels, adj):
    deg = get_degree_matrix(adj)
    for node in range(adj.shape[0]):
        neighbour = torch.nonzero(adj[node]).transpose(0,1)
        nn_deg = [int(deg[nn][nn].item()) for nn in neighbour[0]]
        maxi = np.max(nn_deg)
        if maxi < 17:
            labels[node] = torch.tensor(0)
        else:
            labels[node] = torch.tensor(1)
    return labels

class DataLoader:
    def __init__(self, dataset, self_loops=False):
        self.data = dataset
        self.self_loops = self_loops

    def loadData(self):
        file = open('./data/gnn_explainer/{}.pickle'.format(self.data[:4]), 'rb')
        data = pickle.load(file)
        file.close()
        
        return data

    def preprocessData(self):
        data = self.loadData()
        adj = torch.Tensor(data["adj"]).squeeze()       # Does not include self loops
        if(self.self_loops): #add self loops
            adj.fill_diagonal_(1)
        features = torch.Tensor(data["feat"]).squeeze()
        # print("from data: ", features.shape, features[675]) - all 1s - dim 10
        labels = torch.tensor(data["labels"]).squeeze()
        idx_train = torch.tensor(data["train_idx"])
        idx_test = torch.tensor(data["test_idx"])
        # returns tuple(edge_index, edge_attributes)
        edge_index = dense_to_sparse(adj)  # [0]
        norm_adj = normalize_adj(adj) # According to reparam trick from GCN paper
            
        g = Graph(adj, features, labels, idx_train, idx_test, edge_index, norm_adj)
        return g



