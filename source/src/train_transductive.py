import argparse
import pickle
from utils.dataloader import DataLoader
from utils.classificationnet import GCNSynthetic
from torch.autograd import Variable
from utils.player import Player
from utils.env import Env
from utils.utils import *
from utils.policynet import *
from cmd_args import get_options
import numpy as np
import time
import torch

args = get_options()
torch.set_printoptions(precision=8)
if(args.device == 'cuda'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "1" #set according to gpu avalaibility
torch.manual_seed(args.seed)

#no-kl, correct only, non-zero, deg changes
switcher = {'gcn':PolicyNetwork, 'gat':PolicyNetwork2, 'sage':PolicyNetwork3}
res_file = open('./logs/{}/log_train_{}_{}.txt'.format(args.dataset, args.dataset, args.save_prefix), 'w')

#load model to be explained
def loadModel(args, g):
    if(args.saved):
        model = GCNSynthetic(nfeat=g.feats.shape[1], nhid=args.hidden, nout=args.hidden,
                        nclass=len(g.labels.unique()), dropout=args.dropout).to(args.device)
        model.load_state_dict(torch.load(
            "./models/gcn_3layer_{}.pt".format(args.dataset)))
    else: #train model, save it and return
        model = train_model(args, g).to(args.device)

    model.eval()
    output = model(g.feats.to(args.device), g.norm_adj.to(args.device))
    y_pred_orig = torch.argmax(output, dim=1)
    print("y_true counts: {}".format(np.unique(g.labels.cpu().numpy(), return_counts=True)))
    print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.cpu().numpy(),
                                                    return_counts=True)))      # Confirm model is actually doing something
    print("Accuracy: ", accuracy(output, g.labels))
    return model

def log_results(cf_dict):
    final_res = open('./results/{}/{}_{}.pkl'.format(args.dataset, args.dataset, args.save_prefix),'wb')
    pickle.dump(cf_dict, final_res)
    final_res.close()
    
class SingleTrain(object):
    def __init__(self, args):
        # self.globel_number = 1
        self.args = args
        self.dataset = self.args.dataset
        self.maxbudget = self.args.maxbudget
        self.cf_dict = {}
        
        if(self.dataset not in ['syn1', 'syn4', 'syn5']):
            print("Wrong dataset chosen. Terminating ...")
            exit(0)

        # loading data
        data_obj = DataLoader(args.dataset)
        self.graph = data_obj.preprocessData()
    
        # model whose explanation we will generate
        self.model = loadModel(self.args, self.graph)
        
        # training on test set
        if(self.args.train_on_non_zero): #change made
            idx = (self.graph.labels[self.graph.idx_test] > 0).nonzero(as_tuple=False)
            self.targets = self.graph.idx_test[idx]
            if(self.args.verbose):
                print("Number of non zero instances: ", idx.shape[0])
        else:
            self.targets = self.graph.idx_test #difficult 90:91, 20:21  easy 110:111 medium 38:39
        self.players = []
        self.player_idx = []
        i=0
        for t in self.targets:
            p = Player(self.graph, t, self.model, args).cuda()
            if(self.args.train_on_correct_only):
                # print(self.graph.labels[t], p.orig_out)
                if(self.graph.labels[t].to(args.device) == p.orig_out):
                    self.players.append(p)
                    self.player_idx.append(i)
                    i+=1
            else:
                self.players.append(p)
                self.player_idx.append(i)
                i+=1
            
        if(self.args.verbose):
            print("Number of instances: ", len(self.players))
        self.env = Env(self.players, self.args, torch.max(self.graph.labels)+1)
        #policy distribution depends on num
        
    #train function: initialize policy and train target by target
    #train policy till either max edpisodes or till cf_size is not 1
    def train(self):
        for playerid in self.player_idx:
            if(self.args.verbose):
                print("\n---------- Player: {} -----------\n".format(playerid))
            self.policy=switcher[self.args.policynet](self.args,self.env.statedim).cuda() #change - num_actions
            self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.args.rllr)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [300, 600], gamma=0.1, last_epoch=-1)
    
            for episode in range(0,self.args.maxepisodes): #change made
                if(self.args.verbose):
                    print("\n---------- Episode: {} -----------\n".format(episode))
                rewards, logp_actions, p_actions, entropy, done = self.playOneEpisode(playerid=playerid)
                if(not done):
                    loss = self.finishEpisode(rewards, logp_actions, entropy)
                else:
                    break
        return self.cf_dict      
        
    def playOneEpisode(self, playerid):
        self.playerid = playerid
        self.env.reset(playerid)
        rewards, logp_actions, p_actions = [], [], []
        self.states, self.actions = [], []
        entropy_term = 0 
        b = self.maxbudget
        done = False
        
        orig_out = torch.argmax(self.model(self.players[playerid].G_orig.feats.to(self.args.device), self.players[playerid].G_orig.norm_adj.to(self.args.device))[self.players[playerid].G_orig.target_idx]).item() #self.graph.labels[self.players[playerid].target]#
        curr_out = torch.argmax(self.model(self.players[playerid].G_curr.feats.to(self.args.device), self.players[playerid].G_curr.norm_adj.to(self.args.device))[self.players[playerid].G_curr.target_idx]).item()
        
        while (orig_out == curr_out and b>0 and len(self.players[playerid].cand_dict)>=1):
            state = self.env.getState(playerid) 
            self.states.append(state)
          
            _, logits = self.policy(state[0].to(self.args.device), state[1].to(self.args.device), self.players[playerid].cand_dict)
            action,logp_action, p_action = self.policy.get_action(logits, self.players[playerid].cand_dict)
           
            entropy = -1*(torch.mean(logits) * torch.log(logits)).sum()
            if(self.args.verbose):
                print("P_action: ", p_action)
            logp_actions.append(logp_action)
            p_actions.append(p_action)
            entropy_term += entropy
            #get all loss values here from player functions, send action idx
            reward, loss_pred, loss_graph_dist = self.env.step(action,playerid) 
            rewards.append(reward) 
            if(self.args.verbose):
                print("loss total(reward): ", reward)
                print("loss pred: ", loss_pred)
                print("loss graph dist: ", loss_graph_dist)

            curr_out = self.model(self.players[playerid].G_curr.feats.to(self.args.device), self.players[playerid].G_curr.norm_adj.to(self.args.device))[self.players[playerid].G_curr.target_idx]
            curr_out = torch.argmax(curr_out).item()
            b-=1
        
        if(orig_out != curr_out):
            ## dump node_idx, new_idx, "cf_adj", "sub_adj", "y_pred_cf", "y_pred_orig",
            # "label", "num_nodes", "node_dict"
            if(self.players[playerid].cf == None):
                self.players[playerid].cf = self.players[playerid].cf_cand
            elif(len(self.players[playerid].cf_cand) < len(self.players[playerid].cf)):
                self.players[playerid].cf = self.players[playerid].cf_cand
            
            if(len(self.players[playerid].cf) == 1):
                done = True

            counterfactual = []
            for i in range(len(self.players[playerid].cf)): #made changes
                val = [self.players[playerid].G_curr.reverse_map[self.players[playerid].cf[i][0][0]], self.players[playerid].G_curr.reverse_map[self.players[playerid].cf[i][0][1]], self.players[playerid].cf[i][1]]
                counterfactual.append(val)
            
            res_file.write('target: {}, cf: {}\n'.format(self.players[playerid].target.item(), counterfactual)) #self.players[playerid].cf
            self.cf_dict[self.players[playerid].target.item()] = [self.players[playerid].G_curr.target_idx.item(),  self.players[playerid].G_curr.adj, self.players[playerid].G_orig.adj, counterfactual, self.players[playerid].G_curr.adj.shape[0], self.players[playerid].G_curr.node_map, orig_out, curr_out, self.players[playerid].G.labels[self.players[playerid].target], reward, loss_pred, loss_graph_dist, 'found'] #self.players[playerid].cf
        else:
            counterfactual = []
            for i in range(len(self.players[playerid].cf_cand)): #made changes
                val = [self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][0]], self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][1]], self.players[playerid].cf_cand[i][1]]
                counterfactual.append(val)
            
            if(self.players[playerid].target.item() not in self.cf_dict.keys()):
                self.cf_dict[self.players[playerid].target.item()] = [self.players[playerid].G_curr.target_idx.item(),  self.players[playerid].G_curr.adj, self.players[playerid].G_orig.adj, counterfactual , self.players[playerid].G_curr.adj.shape[0], self.players[playerid].G_curr.node_map, orig_out, curr_out, self.players[playerid].G.labels[self.players[playerid].target], reward, loss_pred, loss_graph_dist, 'not found'] #self.players[playerid].cf
            res_file.write('target: {}, cf: {}\n'.format(self.players[playerid].target.item(), [])) 
            
        logp_actions = torch.stack(logp_actions)
        p_actions = torch.stack(p_actions)
        rewards = torch.stack(rewards)
    
        return rewards,logp_actions, p_actions, entropy_term, done
        
    def finishEpisode(self,rewards, log_probs, entropy): #update policy 
        if(self.args.verbose): 
            print("rewards: ", rewards)
            print("Before squeeze: ", log_probs.shape)
        log_probs = log_probs.squeeze(1)
        if(self.args.verbose): 
            print("After squeeze: ", log_probs.shape)
        loss = rewards.sum() #sum, for cumulative rewards of batch which should be minimized
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.args.discount**pw * (1/r)
                pw = pw + 1
               
        discounted_rewards = np.append(discounted_rewards,  Gt.cpu().detach().numpy()) #-t or 1/Gt
       
        #in order to deal with the case when discounted rewards has just one value, so that resulting gradient does not become 0
        if(len(discounted_rewards) > 1): 
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9) # normalize discounted rewards
        
        vals = torch.tensor(discounted_rewards).to(args.device) 
        
        actor_loss = (-log_probs * vals).mean()
            
        policy_gradient = actor_loss + self.args.ent*entropy
        if(self.args.verbose): 
            print("discounted reward: ", discounted_rewards)
            print("policy loss: ", actor_loss)
            print("policy gradient: ", policy_gradient)

        self.opt.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            policy_gradient.backward()
            self.opt.step()

        return policy_gradient, loss

if __name__ == "__main__":
    singletrain = SingleTrain(args)
    start = time.time()
    cf_dict = singletrain.train()
    time_taken = time.time() - start
    print('Time taken (in sec): ', time_taken)
    print('Time taken (in hours): ', time_taken/3600)
    log_results(cf_dict)
    res_file.close()
