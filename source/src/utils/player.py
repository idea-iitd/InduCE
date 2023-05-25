# individual player who takes the action and evaluates the effect
import pty
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
import math

from utils.classificationnet import GCNSynthetic
from utils.utils import *


class Player(nn.Module):
    def __init__(self,G,t,model,args):
        super(Player,self).__init__()
        print("Target: ", t)
        self.G = G
        self.target = t
        self.args = args
        self.cf_cand = []
        self.cf = None #changes for transductive setting
        self.net = model.to(self.args.device)
        model.eval()

        self.G_orig, self.allnodes_output, self.orig_out, sub_adj, sub_feat, sub_labels, node_dict, new_idx, rev_idx_dict = self.setup()
        self.G_last = Subgraph(self.G_orig.adj.detach().clone(), self.G_orig.feats.detach().clone(), self.G_orig.labels.detach().clone(), self.G_orig.node_map, self.G_orig.target_idx, self.G_orig.reverse_map)
        self.G_curr = Subgraph(self.G_orig.adj.detach().clone(), self.G_orig.feats.detach().clone(), self.G_orig.labels.detach().clone(), self.G_orig.node_map, self.G_orig.target_idx, self.G_orig.reverse_map)
        self.maxbudget = args.maxbudget
        self.reward_func=F.nll_loss
        self.cand_dict = self.getCandidates() 
       
    def setup(self):
        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(
        int(self.target), self.G.edge_index, self.args.k, self.G.feats, self.G.labels)
        new_idx = node_dict[int(self.target)]
        rev_idx_dict = {}
        for key in node_dict:
            rev_idx_dict[node_dict[key]] = key
        # Check that original model gives same prediction on full graph and subgraph
        with torch.no_grad():
            output = self.net(self.G.feats.to(self.args.device), self.G.norm_adj.to(self.args.device))
            print("Output original model, full adj: {}".format(output[self.target]))
            out_sub = self.net(sub_feat.to(self.args.device), normalize_adj(sub_adj).to(self.args.device))
            print("Output original model, sub adj: {}".format(out_sub[new_idx]))
          
        g = Subgraph(sub_adj, sub_feat, sub_labels, node_dict, new_idx, rev_idx_dict)
        out = torch.argmax(out_sub[new_idx])
        return g, out_sub, out, sub_adj, sub_feat, sub_labels, node_dict, new_idx, rev_idx_dict

    def getCandidates(self):
        cand_dict = [] 
        tar = self.G_curr.target_idx

        if(self.args.tar_only):
            for j in range(self.G_curr.adj.shape[0]):
                if(tar != j):
                    #1 means deletion, 0 means addition
                    if(self.args.del_only): #bug
                        if(self.G_curr.adj[tar][j].item() ==  1):
                            cand_dict.append((tar,j,torch.tensor([self.G_curr.adj[tar][j].item()]).to(self.args.device)))
                    else:
                        cand_dict.append((tar,j,torch.tensor([self.G_curr.adj[tar][j].item()]).to(self.args.device)))
        else: 
            # additions from target only
            if(self.args.del_only == False):
                for j in range(self.G_curr.adj.shape[0]):
                    if(tar != j and self.G_curr.adj[tar][j].item() ==  0):
                        cand_dict.append((tar,j,torch.tensor([self.G_curr.adj[tar][j].item()]).to(self.args.device)))
            
            # all possible deletions
            for j in range(self.G_curr.adj.shape[0]):
                k_start = 0
                if(self.args.is_directed == False):
                    k_start = j+1
                for k in range(k_start,self.G_curr.adj.shape[1]):
                    if(j!=k and self.G_curr.adj[j][k].item() ==  1):
                        cand_dict.append((j,k,torch.tensor([self.G_curr.adj[tar][j].item()]).to(self.args.device)))            

        return cand_dict

    def perturb(self,action): #perform action 
        self.G_last =  Subgraph(self.G_curr.adj.detach().clone(), self.G_curr.feats.detach().clone(), self.G_curr.labels.detach().clone(), self.G_curr.node_map, self.G_curr.target_idx, self.G_curr.reverse_map)
        u = self.cand_dict[action][0]
        v = self.cand_dict[action][1]
        p_type = ''
        if(self.G_curr.adj[u][v].item() == 1):
            p_type = 'del'
            self.G_curr.adj[u][v] = torch.tensor(0) 
            if(self.args.is_directed == False):
                self.G_curr.adj[v][u] = torch.tensor(0)
        else:
            p_type = 'add'
            self.G_curr.adj[u][v] = torch.tensor(1) 
            if(self.args.is_directed == False):
                self.G_curr.adj[v][u] = torch.tensor(1)
        
        self.G_curr.norm_adj = normalize_adj(self.G_curr.adj)
        self.G_curr.deg = get_degree_matrix(self.G_curr.adj) #found error
        self.cf_cand.append((self.cand_dict[action],p_type))
        del self.cand_dict[action]
    
    def scaledReward(self, out_probs_curr,  out_probs_last , out_last, out_curr):
        pred_same = (out_curr == out_last).float()
        out_probs_curr = out_probs_curr
        out_last = out_last.unsqueeze(0)
        
        loss_pred = - F.nll_loss(out_probs_curr, out_last.long())
        loss_pred = torch.exp(loss_pred)
        loss_graph_dist = sum(sum(abs(self.G_orig_adj - self.G_curr_adj))) / 2      # Number of edges changed (symmetrical)
        loss_graph_dist = 1/(1+torch.exp(self.args.delta*loss_graph_dist))

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + loss_graph_dist
        # print("Total loss: ", loss_total)
        return loss_total, loss_pred, loss_graph_dist

    def loss(self, out_probs_curr,  out_probs_last , out_last, out_curr):
        pred_same = (out_curr == out_last).float()

        # Need dim >=2 for F.nll_loss to work
        out_probs_curr = out_probs_curr.unsqueeze(0)
        out_last = out_last.unsqueeze(0)
        
        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred =  - F.nll_loss(out_probs_curr, out_last.long())
        
        loss_graph_dist = sum(sum(abs(self.G_orig.adj - self.G_curr.adj))) / 2      # Number of edges changed (symmetrical)

        curr_beta = self.args.beta

        if(self.args.adaptive_beta): #change made  (and loss_graph_dist>5)
            curr_beta /= math.sqrt(loss_graph_dist)
        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + curr_beta * loss_graph_dist
       
        return loss_total, loss_pred, loss_graph_dist


    def oneStepReward(self): #compute reward
        out_probs_last = self.net(self.G_last.feats.to(self.args.device), self.G_last.norm_adj.to(self.args.device))
        out_probs_curr = self.net(self.G_curr.feats.to(self.args.device), self.G_curr.norm_adj.to(self.args.device))
        out_last = torch.argmax(out_probs_last[self.G_last.target_idx])
        out_curr = torch.argmax(out_probs_curr[self.G_curr.target_idx])
        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        if(not self.args.scaled_rewards):
            loss_total, loss_pred, loss_graph_dist = self.loss(out_probs_curr[self.G_curr.target_idx],  out_probs_last[self.G_last.target_idx], out_last, out_curr)
        else:
            loss_total, loss_pred, loss_graph_dist = self.scaledReward(out_probs_curr[self.G_curr.target_idx],  out_probs_last[self.G_last.target_idx], out_last, out_curr)
      
        return loss_total, loss_pred, loss_graph_dist 

    def reset(self):
        self.opt = torch.optim.Adam(self.net.parameters(),lr=self.args.lr,weight_decay=5e-4)
        self.allnodes_output = self.net(self.G_orig.feats.cuda(),self.G_orig.norm_adj.cuda()).detach()
        self.G_last = Subgraph(self.G_orig.adj.detach().clone(), self.G_orig.feats.detach().clone(), self.G_orig.labels.detach().clone(), self.G_orig.node_map, self.G_orig.target_idx, self.G_orig.reverse_map)
        self.G_curr = Subgraph(self.G_orig.adj.detach().clone(), self.G_orig.feats.detach().clone(), self.G_orig.labels.detach().clone(), self.G_orig.node_map, self.G_orig.target_idx, self.G_orig.reverse_map)
        self.cand_dict = self.getCandidates() 
        self.cf_cand = [] 
       