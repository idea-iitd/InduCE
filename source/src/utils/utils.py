from math import floor
import os
import errno
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph
import torch.nn.functional as F
from utils.classificationnet import GCNSynthetic
from torch.utils.data import DataLoader as dl
import os

class Graph:
    def __init__(self, adj, features, labels, idx_train, idx_test, edge_index, norm_adj):
        self.adj = adj
        self.feats = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.edge_index = edge_index
        self.norm_adj = norm_adj

class Subgraph:
    def __init__(self, sub_adj, sub_feat, sub_labels, node_dict, new_idx, rev_idx_dict):
        self.adj = sub_adj
        self.norm_adj = normalize_adj(sub_adj)
        self.feats = sub_feat
        self.labels = sub_labels
        self.node_map = node_dict
        self.reverse_map = rev_idx_dict
        self.target_idx = new_idx
        self.deg = get_degree_matrix(self.adj)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_degree_matrix(adj):
    return torch.diag(sum(adj))


def normalize_adj(adj):
    # Normalize adjacancy matrix according to reparam trick in GCN paper
    A_tilde = adj + torch.eye(adj.shape[0])
    D_tilde = get_degree_matrix(A_tilde)
    # Raise to power -1/2, set all infs to 0s
    D_tilde_exp = D_tilde ** (-1 / 2)
    D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

    # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
    norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
    return norm_adj


def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
    edge_subset = k_hop_subgraph(
        node_idx, n_hops, edge_index[0])     # Get all nodes involved
    # Get relabelled subset of edges
    edge_subset_relabel = subgraph(
        edge_subset[0], edge_index[0], relabel_nodes=True)
    sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
    sub_feat = features[edge_subset[0], :]
    sub_labels = labels[edge_subset[0]]
    new_index = np.array([i for i in range(len(edge_subset[0]))])
    # Maps orig labels to new
    node_dict = dict(zip(edge_subset[0].numpy(), new_index))
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))   
    return sub_adj, sub_feat, sub_labels, node_dict

def train_model(args, g):
    from collections import deque
    # from graph sage get embeddings
    model =GCNSynthetic(nfeat=g.feats.shape[1], nhid=args.hidden, nout=args.hidden,
                            nclass=len(g.labels.unique()), dropout=args.dropout).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # loader = DataLoader(dataset, batch_size=32, shuffle=True)

    feats= g.feats.to(args.device)
    adj = normalize_adj(g.adj).to(args.device)
    idx_train = g.idx_train[:floor(0.8*g.idx_train.shape[0])].to(args.device)
    idx_val = g.idx_train[floor(0.8*g.idx_train.shape[0]):].to(args.device)
    idx_test = g.idx_test.to(args.device)
    labels = g.labels.to(args.device)

    def accuracy(output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def train(epoch, val=True):
        model.train()
        optimizer.zero_grad()
        output = model(feats, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if val:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(feats, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()))


    def test():
        model.eval()
        output = model(feats, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))


    # Train model
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    # Testing
    test()

    PATH = "./models/gcn_3layer_{}.pt".format(args.dataset)
    torch.save(model.state_dict(), PATH)

    return model
  
  