'''
Ioffe, S. and Szegedy, C., 2015, June. Batch normalization: Accelerating deep network training by reducing internal covariate shift. 
In International conference on machine learning (pp. 448-456). PMLR.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv, SAGEConv
import numpy as np
import math
#from src.utils.const import MIN_EPSILON,MAX_EXP
torch.set_printoptions(precision=8)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

#class MLP
class MLP(nn.Module):
	def __init__(self, input_dim, output_dim, num_layers, hidden_dim):

		assert num_layers >= 0 , "invalid input"
		super(MLP, self).__init__()
		layer_sizes = [input_dim] + [hidden_dim]*(num_layers) + [output_dim]
		self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)])
		self.non_linearity = torch.nn.ReLU


	def forward(self, x):
		for i,linear_tranform in enumerate(self.layers):
			x = linear_tranform(x) #torch.clamp_min(linear_tranform(x),1e-14)
			if i!= len(self.layers) - 1:
				x = F.leaky_relu(x) 
		return x

#Policy network 
class PolicyNetwork3(nn.Module):
    def __init__(self, args, statedim, cand_dict=None, learning_rate=3e-4):
        super(PolicyNetwork3, self).__init__()
        self.args = args
        self.candidates = cand_dict
        self.gcn = nn.ModuleList()
        # added batching
        for i in range(self.args.layers):
            if (i == 0):
                self.gcn.append(SAGEConv(statedim, self.args.pnhid, bias=True))
            else:
                self.gcn.append(SAGEConv(self.args.pnhid, self.args.pnhid, bias=True))
            self.gcn.append(nn.BatchNorm1d(self.args.pnhid))

        self.confidence_model = MLP(input_dim = 2*self.args.pnhid+1, output_dim = 1, num_layers = self.args.cfnum_layers, hidden_dim = self.args.pnhid//2)

    def get_candidate_embs(self,x, cand_dict):
        embs = []
        for i in range(len(cand_dict)): #.keys()
            u, v = cand_dict[i][0], cand_dict[i][1]
            embs.append(torch.cat((x[u],x[v],cand_dict[i][2]),-1).cpu().detach().numpy()) #.cpu().detach().numpy() 
        
        return torch.tensor(np.array(embs)).to(self.args.device)

    def forward(self, x, adj, cand_dict=None):
        for i, layer in enumerate(self.gcn):
            if(i%2 == 1):
                x = F.leaky_relu(layer(x))
                x = F.dropout(x, self.args.pdropout, training=self.training)
            else:
                x = layer(x, adj)   
        if(x.isnan().any()):
            x = torch.nan_to_num(x, nan=1e-14) 
        cand_embs = self.get_candidate_embs(x, cand_dict)
        y = self.confidence_model(cand_embs)
        return y, F.softmax(y, dim=0)
    
    def softmax(self, x):
        x = torch.exp(x)/torch.sum(torch.exp(x))
        return x

    def get_action(self, probs, cand_dict=None, train=True):
        if(train):  
            m = torch.distributions.Categorical(probs.reshape(1,-1))
            highest_prob_action = m.sample()
            prob = probs.squeeze(0)[highest_prob_action]
            log_prob = m.log_prob(highest_prob_action)
        else:
            highest_prob_action = np.argmax(probs.cpu().detach().numpy())
            prob = probs.squeeze(0)[highest_prob_action]
            log_prob = torch.log(prob)
        return highest_prob_action, log_prob, prob

#Policy network 
class PolicyNetwork2(nn.Module):
    def __init__(self, args, statedim, cand_dict=None, learning_rate=3e-4):
        super(PolicyNetwork2, self).__init__()
        self.args = args
        self.candidates = cand_dict
        self.gcn = nn.ModuleList()
        # added batching
        for i in range(self.args.layers):
            if (i == 0):
                self.gcn.append(GATConv(statedim, self.args.pnhid, bias=True))
            else:
                self.gcn.append(GATConv(self.args.pnhid, self.args.pnhid, bias=True))
            self.gcn.append(nn.BatchNorm1d(self.args.pnhid))

        self.confidence_model = MLP(input_dim = 2*self.args.pnhid+1, output_dim = 1, num_layers = self.args.cfnum_layers, hidden_dim = self.args.pnhid//2)

    def get_candidate_embs(self,x, cand_dict):
        embs = []
        for i in range(len(cand_dict)): #.keys()
            u, v = cand_dict[i][0], cand_dict[i][1]
            embs.append(torch.cat((x[u],x[v],cand_dict[i][2]),-1).cpu().detach().numpy()) #.cpu().detach().numpy() 
        
        return torch.tensor(np.array(embs)).to(self.args.device)

    def forward(self, x, adj, cand_dict=None):
        for i, layer in enumerate(self.gcn):
            if(i%2 == 1):
                x = F.leaky_relu(layer(x))
                x = F.dropout(x, self.args.pdropout, training=self.training)
            else:
                x = layer(x, adj)   
        if(x.isnan().any()):
            x = torch.nan_to_num(x, nan=1e-14) 
        cand_embs = self.get_candidate_embs(x, cand_dict)
        y = self.confidence_model(cand_embs)
        return y, F.softmax(y, dim=0)
    
    def softmax(self, x):
        x = torch.exp(x)/torch.sum(torch.exp(x))
        return x

    def get_action(self, probs, cand_dict=None, train=True):
        if(self.args.verbose):
            print("action probs\n", probs)
        if(train):  
            m = torch.distributions.Categorical(probs.reshape(1,-1))
            highest_prob_action = m.sample()
            prob = probs[highest_prob_action] #.squeeze(0) made change for bug
            log_prob = m.log_prob(highest_prob_action)
        else:
            highest_prob_action = np.argmax(probs.cpu().detach().numpy())
            prob = probs.squeeze(0)[highest_prob_action]
            log_prob = torch.log(prob)
        if(self.args.verbose):
            print("From policy: \nChosen action: {}, prob: {}, log_prob, {}".format(highest_prob_action, prob, log_prob))
        return highest_prob_action, log_prob, prob

#Policy network 
class PolicyNetwork(nn.Module):
    def __init__(self, args, statedim, cand_dict=None, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()
        self.args = args
        self.candidates = cand_dict
        self.gcn = nn.ModuleList()
        # added batching
        for i in range(self.args.layers):
            if (i == 0):
                self.gcn.append(GraphConvolution(statedim, self.args.pnhid, bias=True))
            else:
                self.gcn.append(GraphConvolution(self.args.pnhid, self.args.pnhid, bias=True))
            self.gcn.append(nn.BatchNorm1d(self.args.pnhid))

        self.confidence_model = MLP(input_dim = 2*self.args.pnhid+1, output_dim = 1, num_layers = self.args.cfnum_layers, hidden_dim = self.args.pnhid//2)

    def get_candidate_embs(self,x, cand_dict):
        embs = []
        for i in range(len(cand_dict)): #.keys()
            u, v = cand_dict[i][0], cand_dict[i][1]
            embs.append(torch.cat((x[u],x[v],cand_dict[i][2]),-1).cpu().detach().numpy()) #.cpu().detach().numpy() 
        
        return torch.tensor(np.array(embs)).to(self.args.device)

    def forward(self, x, adj, cand_dict=None):
        for i, layer in enumerate(self.gcn):
            if(i%2 == 1):
                x = F.leaky_relu(layer(x))
                x = F.dropout(x, self.args.pdropout, training=self.training)
            else:
                x = layer(x, adj)

        if(x.isnan().any()):
            x = torch.nan_to_num(x, nan=1e-14) 
        cand_embs = self.get_candidate_embs(x, cand_dict)
        y = self.confidence_model(cand_embs)
        return y, F.softmax(y, dim=0)
    
    def softmax(self, x):
        x = torch.exp(x)/torch.sum(torch.exp(x))
        return x

    def get_action(self, probs, cand_dict=None, train=True):
        #sample - train, test - not sample
        if(train):  
            m = torch.distributions.Categorical(probs.reshape(1,-1))
            highest_prob_action = m.sample()
            prob = probs.squeeze(0)[highest_prob_action]
            log_prob = m.log_prob(highest_prob_action)
        else:
            highest_prob_action = np.argmax(probs.cpu().detach().numpy())
            prob = probs.squeeze(0)[highest_prob_action]
            log_prob = torch.log(prob)
        return highest_prob_action, log_prob, prob

