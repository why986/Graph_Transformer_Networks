#!/usr/bin/env python
# coding: utf-8
# pylint: disable = E1101
# pylint: disable = C0103, C0114, C0115, C0116
# pylint: disable = W0621

import pickle
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

n_epoch = 40
node_dim = 64
num_channels = 2
learning_rate = 0.005
weight_decay = 0.001
num_layers = 2
flag_norm = True
adaptive_lr = True
dataset_name = 'ACM'

with open('data/' + dataset_name + '/node_features.pkl', 'rb') as f:
    node_features = pickle.load(f)
with open('data/' + dataset_name + '/edges.pkl', 'rb') as f:
    edges = pickle.load(f)
with open('data/' + dataset_name + '/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
num_nodes = edges[0].shape[0]
node_features = torch.from_numpy(node_features).type(torch.FloatTensor)

train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.LongTensor)
train_label = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.LongTensor)
valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.LongTensor)
valid_label = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.LongTensor)
test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.LongTensor)
test_label = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.LongTensor)

for i, edge in enumerate(edges):
    if i == 0:
        A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
    else:
        A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)]
                      , dim=-1)
A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

num_classes = int(torch.max(train_label).item() + 1)
num_edges = len(edges) + 1
A = A.unsqueeze(0).permute(0, 3, 1, 2)


class GTN(nn.Module):
    def __init__(self, num_edges, num_channels, num_layers,
                 input_shape, node_dim, output_shape, flag_norm):
        super(GTN, self).__init__()
        self.num_edges = num_edges
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.node_dim = node_dim
        self.output_shape = output_shape
        self.flag_norm = flag_norm
        layers = []
        for i in range(num_layers):
            layers.append(GTLayer(num_edges, num_channels, flag_first=(i == 0)))
        self.layers = nn.ModuleList(layers)

        self.weight = nn.Parameter(torch.Tensor(input_shape, node_dim))

        self.linear1 = nn.Linear(node_dim * num_channels, node_dim)
        self.linear2 = nn.Linear(node_dim, output_shape)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i == 0:
                retH = self.norm(H[i, :, :]).unsqueeze(0)
            else:
                retH = torch.cat((retH, self.norm(H[i, :, :]).unsqueeze(0)), dim=0)
        return retH

    def norm(self, H):
        if not self.flag_norm:
            return H
        H = H.t()
        H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor)) \
            + torch.eye(H.shape[0]).type(torch.FloatTensor)
        D = torch.sum(H, dim=1)
        D_inv = D.pow(-1)
        D_inv[D_inv == float('inf')] = 0
        D_inv = D_inv * torch.eye(H.shape[0]).type(torch.FloatTensor)
        H = torch.mm(D_inv, H)
        H = H.t()
        return H

    def conv(self, X, H):
        X = torch.mm(X, self.weight)
        H = self.norm(H)
        return torch.mm(H.t(), X)

    def forward(self, A, features, nodes):
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)

        for i in range(self.num_channels):
            if i == 0:
                X = nn.functional.relu(self.conv(features, H[i]))
            else:
                tX = nn.functional.relu(self.conv(features, H[i]))
                X = torch.cat((X, tX), dim=1)
        Y = self.linear1(X)
        Y = nn.functional.relu(Y)
        predict = self.linear2(Y[nodes])
        return predict, Ws


class GTLayer(nn.Module):
    def __init__(self, num_edges, num_channels, flag_first):
        super(GTLayer, self).__init__()
        self.num_edges = num_edges
        self.num_channels = num_channels
        self.flag_first = flag_first
        self.W1 = nn.Parameter(torch.Tensor(num_channels, num_edges, 1, 1))
        if flag_first:
            self.W2 = nn.Parameter(torch.Tensor(num_channels, num_edges, 1, 1))

    def reset_parameters(self):
        nn.init.constant_(self.W1, 0.1)
        if self.flag_first:
            nn.init.constant_(self.W2, 0.1)

    def forward(self, A, _H=None):
        if self.flag_first:
            _W1 = nn.functional.softmax(self.W1, dim=1).detach()
            _W2 = nn.functional.softmax(self.W2, dim=1).detach()
            Q1 = torch.sum(A * _W1, dim=1)
            Q2 = torch.sum(A * _W2, dim=1)
            A_ = torch.bmm(Q1, Q2)
            W = [_W1, _W2]
        else:
            _W1 = nn.functional.softmax(self.W1, dim=1).detach()
            Q1 = torch.sum(A * _W1, dim=1)
            A_ = torch.bmm(_H, Q1)
            W = [_W1]
        return A_, W


model = GTN(num_edges=num_edges,
            num_channels=num_channels,
            num_layers=num_layers,
            input_shape=node_features.shape[1],
            node_dim=node_dim,
            output_shape=num_classes,
            flag_norm=flag_norm)
if adaptive_lr:
    optimizer = torch.optim.Adam([{'params': model.weight},
                                  {'params': model.linear1.parameters()},
                                  {'params': model.linear2.parameters()},
                                  {'params': model.layers.parameters(), "lr": 0.5}],
                                 lr=learning_rate, weight_decay=weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()


def train(model, train_node, train_label, optimizer, criterion):
    model.zero_grad()
    model.train()
    predict, _ = model(A, node_features, train_node)
    loss = criterion(predict, train_label)
    loss.backward()
    optimizer.step()
    predict = torch.argmax(predict, dim=1)
    acc = (predict == train_label).sum().item() / train_label.numel()
    F_score = f1_score(predict.detach(), train_label, average='macro').item()
    return loss, acc, F_score


def test(model, test_node, test_label, criterion):
    model.eval()
    with torch.no_grad():
        predict, _ = model(A, node_features, test_node)
        loss = criterion(predict, test_label)
        predict = torch.argmax(predict, dim=1)
        acc = (predict == test_label).sum().item() / test_label.numel()
        F_score = f1_score(predict.detach(), test_label, average='macro').item()
    return loss, acc, F_score


best_f1 = 0.0

for epoch in range(n_epoch):
    starttime = time.time()

    for param_group in optimizer.param_groups:
        if param_group['lr'] > learning_rate:
            param_group['lr'] = param_group['lr'] * 0.9

    train_loss, train_acc, train_f_score = \
        train(model, train_node, train_label, optimizer, criterion)
    valid_loss, valid_acc, valid_f_score = \
        test(model, valid_node, valid_label, criterion)
    test_loss, test_acc, test_f_score = \
        test(model, test_node, test_label, criterion)

    endtime = time.time()
    epoch_mins = int((endtime - starttime) / 60)
    epoch_secs = int(endtime - starttime - epoch_mins * 60)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t Train Loss: {train_loss:.5f} | Train Acc: {train_acc * 100:.3f}% '
          f'| Train F score: {train_f_score:.3f}')
    print(f'\t Validation Loss: {valid_loss:.5f} | Validation Acc: {valid_acc * 100:.3f}% '
          f'| Validation F score: {valid_f_score:.3f}')
    print(f'\t Test Loss: {test_loss:.5f} | Test Acc: {test_acc * 100:.3f}% '
          f'| Test F score: {test_f_score:.3f}')
    with open("log.txt", "a") as f:
        f.write(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
        f.write(f'\t Train Loss: {train_loss:.5f} | Train Acc: {train_acc * 100:.3f}% '
                f'| Train F score: {train_f_score:.3f}\n')
        f.write(f'\t Validation Loss: {valid_loss:.5f} | Validation Acc: {valid_acc * 100:.3f}% '
                f'| Validation F score: {valid_f_score:.3f}\n')
        f.write(f'\t Test Loss: {test_loss:.5f} | Test Acc: {test_acc * 100:.3f}% '
                f'| Test F score: {test_f_score:.3f}\n')

    if best_f1 < valid_f_score:
        best_f1 = valid_f_score
        print(test_f_score)
        torch.save(model.state_dict(), 'model_5.pt')
