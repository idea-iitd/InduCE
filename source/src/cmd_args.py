import argparse
import os
import torch
import time

def get_options(args=None):
    parser = argparse.ArgumentParser()
    #save name prefix
    parser.add_argument('--save_prefix', default='', help='add prefix name for separately saving results of variation in experiments')
    #data related params
    parser.add_argument('--dataset', default='syn1', help='syn1/syn4/syn5')
    parser.add_argument('--is_directed', default=False)
    parser.add_argument('--train_on_correct_only', action='store_true', help='use only those nodes that have correct prediction for training')
    parser.add_argument('--train_on_non_zero', action='store_true', help='use only non zero label nodes for training')
    parser.add_argument('--add_self_loops', action='store_true', help='check GAT policy performance with self loop')

    # Based on original GCN models -- do not change (taken fron cf-gnn paper)
    parser.add_argument('--saved', type=int, default=1,
                        help='1:saved, 0:unsaved')
    parser.add_argument('--hidden', type=int, default=20,
                        help='Number of hidden units.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of convolutional layers.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (between 0 and 1)')
    parser.add_argument("--lr",type=float,default=1e-2)

    # arguments for RL model
    parser.add_argument('--maxbudget', type=int, default=10,
                        help='integer')
    parser.add_argument('--maxepisodes', type=int, default=1000,
                        help='integer')
    parser.add_argument('--k', type=int, default=3,
                        help='k-hop neighbourhood - integer')
    parser.add_argument('--discount', type=float, default=0.6,
                        help='Discount factor (between 0 and 1)')
    parser.add_argument("--rllr",type=float,default=3e-4)
    parser.add_argument("--beta",type=float,default=0.5)
    parser.add_argument("--delta",type=float,default=0.01,help='scaling factor for distance loss for scaled rewards')
    parser.add_argument('--del_only', action='store_true', help='Set this value to only use deletions')
    parser.add_argument('--tar_only', action='store_true', help='Set this value to only use deletions and additions from target to neighbour')
    parser.add_argument('--adaptive_beta', action='store_true', help='Set this value to make beta adaptive')
    parser.add_argument('--scaled_rewards', action='store_true', help='Set this value to use alternate reward function with scaled values of both l_pred and l_dist')
    
    # arguments for policy
    parser.add_argument("--pnhid",type=int,default=16)
    parser.add_argument("--layers",type=int,default=3)
    parser.add_argument("--pdropout",type=float,default=0.0)
    parser.add_argument("--batch_size",type=int,default=2)#changed
    parser.add_argument("--policynet",type=str,default='gcn')
    parser.add_argument("--cfnum_layers",type=int,default=2)

    #arguments for state
    parser.add_argument('--use_entropy', action='store_true', help='use entropy heauristic')
    parser.add_argument('--use_degree', action='store_true', help='use degree heauristic')
    parser.add_argument('--use_onehot', action='store_true', help='use one hot encoded label vector')
    parser.add_argument("--use_node_feats",type=int,default=1)
   
    #others
    parser.add_argument("--ent",type=float,default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda', help='cpu or cuda.')
    parser.add_argument('--verbose', action='store_true', help='Set to print intermediate output')
    
    args = parser.parse_args()
    print(args)
    return args