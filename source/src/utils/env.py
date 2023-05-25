import  torch.multiprocessing as mp
import time
import torch
import torch.nn.functional as F
import numpy as np

from utils.player import Player
from utils.utils import *


def logprob2Prob(logprobs,multilabel=False):
    #print(logprobs.shape)
    if multilabel:
        probs = torch.sigmoid(logprobs)
        # print(logprobs[:,0], probs[:,0])
    else:
        probs = F.softmax(logprobs, dim=0)
    return probs

def normalizeEntropy(entro,classnum): #this is needed because different number of classes will have different entropy
    maxentro = np.log(float(classnum))
    entro = entro/maxentro  
    return entro

def prob2Logprob(probs,multilabel=False):
    if multilabel:
        raise NotImplementedError("multilabel for prob2Logprob is not implemented")
    else:
        logprobs = torch.log(probs)
    return logprobs

def perc(input):
    # the biger valueis the biger result is
    numnode = input.size(-2)
    res = torch.argsort(torch.argsort(input, dim=-2), dim=-2) / float(numnode)
    return res

def entropy(probs):
    num_classes = probs.shape[0]
    ent = torch.tensor([-torch.sum(probs[:,i] * torch.log2(probs[:,i])) for i in range(probs.shape[1])])
    return ent

def degprocess(deg):
    degree = torch.tensor([deg[i][i] for i in range(deg.shape[0])])
    return degree #/torch.max(degree) changed for testing

def localdiversity(probs, target, adj,deg):
    classnum = probs.shape[0]
    maxentro = np.log2(float(classnum))
    KL = []
    rev_KL = []

    for i in range(adj.shape[0]):
        neighbour = torch.nonzero(adj[i]).transpose(0,1)
        kl_vu = torch.mean(torch.tensor([torch.sum(probs[:,target] * torch.log(probs[:,target]) - probs[:,target] * torch.log(probs[:,nn])) for nn in neighbour[0]]))
        kl_uv = torch.mean(torch.tensor([torch.sum(probs[:,nn] * torch.log(probs[:,nn]) - probs[:,nn] * torch.log(probs[:,target]))for nn in neighbour[0]]))
        KL.append(kl_vu.item()) 
        rev_KL.append(kl_uv.item())
   
    return torch.tensor(KL), torch.tensor(rev_KL) #normalized


def one_hot(labels, num_classes):
    labels = F.one_hot(labels, num_classes)
    return labels

class Env(object):
    ## an environment for multiple players testing the policy at the same time
    def __init__(self,players,args,num_classes):
        '''
        players: a list containing main player (many task) (or only one task
        '''
        self.players = players
        self.args = args
        self.nplayer = len(self.players)
        self.num_classes = num_classes
        self.graphs = [p.G for p in self.players]
        self.targets = [p.target for p in self.players]
        featdim =-1
        self.statedim = self.getState(0)[0].shape[1]
        if(self.args.verbose):
            print("Dim of state: ", self.statedim)

    def step(self,actions,playerid=0):
        p = self.players[playerid]
        p.perturb(actions)
        reward, loss_pred, loss_graph_dist  = p.oneStepReward() #reward = total_loss
        return reward, loss_pred, loss_graph_dist 


    def getState(self, playerid=0):
        p = self.players[playerid]
        output = logprob2Prob(p.allnodes_output.transpose(0,1), multilabel=True)
        state = self.makeState(self.players[playerid].G_orig.labels, output, p.G_curr.deg, playerid, multilabel=True)
        return state

    def reset(self,playerid=0):
        self.players[playerid].reset()

    def makeState(self, labels, probs, deg, playerid, adj=None, multilabel=False):   
        entro = entropy(probs)
        # entro = normalizeEntropy(entro,probs.shape[0]) ## in order to transfer - changed for testing
        deg = degprocess(deg)
        labels = one_hot(labels, self.num_classes)

        features = []
        
        if self.args.use_entropy:
            features.append(entro)
           
        if self.args.use_degree:
            features.append(deg)

        if(self.args.use_entropy or self.args.use_degree): 
            features = torch.stack(features,dim=0).transpose(0,1)
        # print(features.shape)
        if self.args.use_onehot:
            if(torch.is_tensor(features)): #to handle the case when empty feature list comes till here in ablation study
                features = torch.cat((features, labels), dim=1)
            else:
                features = labels
        
        if self.args.use_node_feats:
            if(torch.is_tensor(features)):
                features = torch.cat((features, self.players[playerid].G_curr.feats), dim=1)
            else:
                features = self.players[playerid].G_curr.feats

        if(self.args.policynet == 'gat' or self.args.policynet == 'sage' or self.args.policynet == 'actorcritic'):
            edge_index = dense_to_sparse(self.players[playerid].G_curr.adj)[0].to(self.args.device)
            state = (features, edge_index)
        else:    
            state = (features, self.players[playerid].G_curr.norm_adj)
        
        return state