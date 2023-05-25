import argparse
from cmd_args import get_options
from utils.dataloader import DataLoader
from utils.classificationnet import GCNSynthetic
from utils.player import Player
from utils.env import Env
from utils.utils import *
from utils.policynet import *
import numpy as np
import time
import torch
import pickle
from cmd_args import get_options

args = get_options()


if(args.device == 'cuda'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "1" #set according to gpu avalaibility

switcher = {'gcn':PolicyNetwork, 'gat':PolicyNetwork2, 'sage':PolicyNetwork3}
res_file = open('./logs/{}/log_test_{}_{}.txt'.format(args.dataset, args.dataset, args.save_prefix), 'w')
# PATH = "./saved_models/model_{}.pt".format(args.dataset)

#load model to be explained
def loadModel(args, g):
    if(args.saved):
        model = GCNSynthetic(nfeat=g.feats.shape[1], nhid=args.hidden, nout=args.hidden,
                        nclass=len(g.labels.unique()), dropout=args.dropout)
        model.load_state_dict(torch.load(
            "./models/gcn_3layer_{}.pt".format(args.dataset)))
    else: #train model, save it and return
        data_obj = DataLoader(args.dataset)
        g = data_obj.preprocessData()
        train_model(args, g)

    model.eval()
    output = model(g.feats, g.norm_adj)
    y_pred_orig = torch.argmax(output, dim=1)
    print("y_true counts: {}".format(np.unique(g.labels.numpy(), return_counts=True)))
    print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(),
                                                    return_counts=True)))      # Confirm model is actually doing something
    print("Accuracy: ", accuracy(output, g.labels))
    return model

def log_results(cf_dict):
    final_res = open('./results/{}/{}_test_{}.pkl'.format(args.dataset, args.dataset, args.save_prefix),'wb') #no_kl_del_only_inductive overwritten by adaptive beta
    pickle.dump(cf_dict, final_res)
    final_res.close()

class Eval(object):
    def __init__(self, args):
        self.args = args
        self.dataset = self.args.dataset
        # loading data
        data_obj = DataLoader(args.dataset)
        self.graph = data_obj.preprocessData()
       
        # model whose explanation we will generate
        self.model = loadModel(self.args, self.graph)

        #explainaing only non-zero and correctly-classified instances
        idx = (self.graph.labels[self.graph.idx_test] > 0).nonzero(as_tuple=False)
        self.targets = self.graph.idx_test[idx] 
        if(self.args.verbose):
                print("Number of non zero instances: ", idx.shape[0])
        
        self.players, self.rshapers = [], []
        self.cf_dict = {}
        self.chosen_targets = []
        i=0
        for t in self.targets:
            p = Player(self.graph, t, self.model, args).cuda()
            if(self.graph.labels[t].to(args.device) == p.orig_out):
                self.players.append(p)
                self.chosen_targets.append(t.item())
                i+=1

        if(self.args.verbose):
            print("Number of instances: ", len(self.players))
            print('Chosen targets (idx): ', self.chosen_targets)  
        #save eval set indices in pkl file
        eval_set = open('./eval_set/{}.pkl'.format(args.dataset),'wb') 
        pickle.dump(self.chosen_targets, eval_set)
        eval_set.close() 

        self.env = Env(self.players, self.args, torch.max(self.graph.labels)+1)
        self.policy = switcher[args.policynet](args,self.env.statedim).cuda()
        self.policy.load_state_dict(torch.load("./saved_models/{}/model_{}_{}.pt".format(args.dataset, args.dataset, args.save_prefix))['model_state_dict'])
       
        
    def policyQuery(self, playerid=0):
        self.playerid = playerid
        self.env.reset(playerid)
        rewards, logp_actions, p_actions = [], [], []
        self.states, self.actions = [], []

        initialrewards=0 #reward at timestep 0
        rewards.append(initialrewards)

        b = self.args.maxbudget 
        orig_out = torch.argmax(self.model(self.players[playerid].G_orig.feats.to(self.args.device), self.players[playerid].G_orig.norm_adj.to(self.args.device))[self.players[playerid].G_orig.target_idx]).item()
        curr_out = torch.argmax(self.model(self.players[playerid].G_curr.feats.to(self.args.device), self.players[playerid].G_curr.norm_adj.to(self.args.device))[self.players[playerid].G_curr.target_idx]).item()
        while (orig_out == curr_out and b>0 and len(self.players[playerid].cand_dict)>1): 
            state = self.env.getState(playerid) #state = (feats, norm_adj)
            self.states.append(state)
            _, logits = self.policy(state[0].to(self.args.device), state[1].to(self.args.device), self.players[playerid].cand_dict)
            action,logp_action, p_action = self.policy.get_action(logits, self.players[playerid].cand_dict, False)
            
            logp_actions.append(logp_action)
            p_actions.append(p_action)
            reward, loss_pred, loss_graph_dist = self.env.step(action,playerid)
            rewards.append(reward) #send action idx
            curr_out = torch.argmax(self.model(self.players[playerid].G_curr.feats.to(self.args.device), self.players[playerid].G_curr.norm_adj.to(self.args.device))[self.players[playerid].G_curr.target_idx]).item()
            b-=1
        
        counterfactual = []
        flag = 'not found'
        if(orig_out != curr_out):
            for i in range(len(self.players[playerid].cf_cand)):
                val = [self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][0]], self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][1]], self.players[playerid].cf_cand[i][1]]
                counterfactual.append(val)
            res_file.write('target: {}, cf: {}\n'.format(self.players[playerid].target.item(), counterfactual)) #self.players[playerid].cf
            # print(self.players[playerid].cf_cand)
            flag = 'found'
        else:
            for i in range(len(self.players[playerid].cf_cand)):
                val = [self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][0]], self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][1]], self.players[playerid].cf_cand[i][1]]
                counterfactual.append(val)


        logp_actions = torch.stack(logp_actions)
        p_actions = torch.stack(p_actions)
        
        return counterfactual, self.players[playerid].G_curr.target_idx.item(),  self.players[playerid].G_curr.adj, self.players[playerid].G_orig.adj, self.players[playerid].G_curr.adj.shape[0], self.players[playerid].G_curr.node_map, orig_out, curr_out, self.players[playerid].G.labels[self.players[playerid].target], reward, loss_pred, loss_graph_dist, flag
    
    def policyQueryRun(self):
        for i, p in enumerate(self.players):
            cf, new_idx, cf_adj, sub_adj, num_nodes, node_dict ,orig_label, cf_label, label, total_loss, loss_pred, loss_graph_dist, found = self.policyQuery(i)
            self.cf_dict[p.target.item()] = [new_idx, cf_adj, sub_adj, cf, num_nodes, node_dict ,orig_label, cf_label, label, total_loss, loss_pred, loss_graph_dist, found]
            
        return self.cf_dict
 

if __name__ == "__main__":
    eval = Eval(args)
    start = time.time()
    cf_dict = eval.policyQueryRun()
    print('Time taken (in sec): ', time.time() - start) 
    log_results(cf_dict)
    res_file.close()