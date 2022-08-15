import argparse, time, os, pickle
import numpy as np

import dgl
import torch
import torch.optim as optim
import pandas as pd

from models import LANDER
from dataset import LanderDataset
from test_fun import eval_test

from utils.utils import split_batch

print("Start Time : ")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
###########
# ArgParser
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--levels', type=str, default='1')
parser.add_argument('--faiss_gpu', action='store_true')
parser.add_argument('--model_filename', type=str, default='lander.pth')

# KNN
parser.add_argument('--knn_k', type=str, default='10')
parser.add_argument('--num_workers', type=int, default=0)

# Model
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--num_conv', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--gat', action='store_true')
parser.add_argument('--gat_k', type=int, default=1)
parser.add_argument('--balance', action='store_true')
parser.add_argument('--use_cluster_feat', action='store_true')
parser.add_argument('--use_focal_loss', action='store_true')

parser.add_argument('--cpu', action='store_true')
parser.add_argument('--consider_batch', action='store_true')

# Training
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-5)

# Testing
parser.add_argument('--test_knn_k', type=str, default='3,5,7,8,9,10')
parser.add_argument('--test_levels', type=str, default='4,3,2,2,2,2')
parser.add_argument('--begin_test_epoch', type=int, default=250)
parser.add_argument('--epoch_eval', type=int, default=1)
parser.add_argument('--test_data', type=str, default='')
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--threshold', type=str, default='prob')
parser.add_argument('--metrics', type=str, default='pairwise,bcubed,nmi,ari')
parser.add_argument('--early_stop', action='store_true')
parser.add_argument('--use_gt', action='store_true')

args = parser.parse_args()
print(args)

###########################
# Environment Configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if args.cpu:
    device = torch.device('cpu')

args.device = device

k_list = [int(k) for k in args.knn_k.split(',')]
lvl_list = [int(l) for l in args.levels.split(',')]
train_dat_list = [str(d) for d in args.train_data.split(',')]

def set_train_sampler_loader(g, k):
    fanouts = [k-1 for i in range(args.num_conv + 1)]
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    # fix the number of edges
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g, torch.arange(g.number_of_nodes()), sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers
    )
    return train_dataloader

##################
# Data Preparation
print("Loading Training Data")
train_loaders_list = []
train_dat_list_batch = []
for dat in range(len(train_dat_list)):
    dat_name = train_dat_list[dat]
    print("Preparing dataset : " + dat_name)
    if args.consider_batch:
        with open(args.data_path + dat_name + ".pkl", 'rb') as f:
            features, labels, batch = pickle.load(f)
    else:
        with open(args.data_path + dat_name + ".pkl", 'rb') as f:
            features, labels = pickle.load(f)

    dat_s_i = []

    n_cells = features.shape[0]
    if n_cells <= args.batch_size:
        features_s_i = [features]
        labels_s_i = [labels]
    else:
        features_s_i, labels_s_i = split_batch(x=features, lbs=labels, batch_size=args.batch_size)

    for idx_s in range(len(features_s_i)):
        if len(features_s_i) == 1:
            dat_name_s = dat_name
        else:
            dat_name_s = dat_name + "-" + str(idx_s + 1) + "/" + str(len(features_s_i))
        dat_s_i.append(dat_name_s)

        features_s = features_s_i[idx_s]
        labels_s = labels_s_i[idx_s]

        gs = []
        nbrs = []
        ks = []
        for k, l in zip(k_list, lvl_list):
            dataset = LanderDataset(features=features_s, labels=labels_s, k=k,
                                    levels=l, faiss_gpu=args.faiss_gpu)
            gs += [g for g in dataset.gs]
            ks += [k for g in dataset.gs]
            nbrs += [nbr for nbr in dataset.nbrs]

        train_loaders = []
        for gidx, g in enumerate(gs):
            train_dataloader = set_train_sampler_loader(gs[gidx], ks[gidx])
            train_loaders.append(train_dataloader)
        train_loaders_list.append(train_loaders)
    train_dat_list_batch = train_dat_list_batch + dat_s_i

if args.test_data != "":
    print("Loading Test Data")
    if args.consider_batch:
        with open(args.data_path + args.test_data + ".pkl", 'rb') as f:
            test_features, test_labels, test_batch = pickle.load(f)
    else:
        with open(args.data_path + args.test_data + ".pkl", 'rb') as f:
            test_features, test_labels = pickle.load(f)

print('Dataset Prepared.')

##################
# Model Definition
feature_dim = gs[0].ndata['features'].shape[1]
model = LANDER(feature_dim=feature_dim, nhid=args.hidden,
               num_conv=args.num_conv, dropout=args.dropout,
               use_GAT=args.gat, K=args.gat_k,
               balance=args.balance,
               use_cluster_feat=args.use_cluster_feat,
               use_focal_loss=args.use_focal_loss)
model = model.to(device)
model.train()

#################
# Hyperparameters
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay)

print('Start Training.')

###############
# Training Loop
if args.test_data != "" and args.consider_batch :
    if len(np.unique(test_batch)) != 1:
        results_out_columns = ["Dataset", "Epoch", "KNN_K", "Levels", "ARI_CT", "NMI_CT", "ARI_BC", "NMI_BC"]
        results_out = pd.DataFrame([{"Dataset": "", "Epoch": 0,
                                     "KNN_K": 0, "Levels": 0,
                                     "ARI_CT": 0.0000, "NMI_CT": 0.0000,
                                     "ARI_BC": 0.0000, "NMI_BC": 0.0000}]).drop(index=0)
        results_out.columns = results_out_columns
    else:
        results_out_columns = ["Dataset", "Epoch", "KNN_K", "Levels", "ARI_CT", "NMI_CT"]
        results_out = pd.DataFrame([{"Dataset": "", "Epoch": 0,
                                     "KNN_K": 0, "Levels": 0,
                                     "ARI_CT": 0.0000, "NMI_CT": 0.0000}]).drop(index=0)
        results_out.columns = results_out_columns

for dat in range(len(train_dat_list_batch)):
    dat_name = train_dat_list_batch[dat]
    print("Training with dataset : " + dat_name)

    train_loaders = train_loaders_list[dat]

    # keep num_batch_per_loader the same for every sub_dataloader
    num_batch_per_loader = len(train_loaders[0])
    train_loaders = [iter(train_loader) for train_loader in train_loaders]
    num_loaders = len(train_loaders)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt,
                                                     T_max=args.epochs * num_batch_per_loader * num_loaders,
                                                     eta_min=1e-5)

    for epoch in range(args.epochs):
        loss_den_val_total = []
        loss_conn_val_total = []
        loss_val_total = []
        for batch in range(num_batch_per_loader):
            for loader_id in range(num_loaders):
                try:
                    minibatch = next(train_loaders[loader_id])
                except:
                    train_loaders[loader_id] = iter(set_train_sampler_loader(gs[loader_id], ks[loader_id]))
                    minibatch = next(train_loaders[loader_id])
                input_nodes, sub_g, bipartites = minibatch
                sub_g = sub_g.to(device)
                bipartites = [b.to(device) for b in bipartites]
                # get the feature for the input_nodes
                opt.zero_grad()
                output_bipartite = model(bipartites)
                loss, loss_den_val, loss_conn_val = model.compute_loss(output_bipartite)
                loss_den_val_total.append(loss_den_val)
                loss_conn_val_total.append(loss_conn_val)
                loss_val_total.append(loss.item())
                loss.backward()
                opt.step()
                if (batch + 1) % 10 == 0:
                    print('epoch: %d, batch: %d / %d, loader_id : %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f'%
                          (epoch, batch, num_batch_per_loader, loader_id, num_loaders,
                           loss.item(), loss_den_val, loss_conn_val))
                scheduler.step()
        print('epoch: %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f'%
              (epoch, np.array(loss_val_total).mean(),
               np.array(loss_den_val_total).mean(), np.array(loss_conn_val_total).mean()))
        torch.save(model.state_dict(), args.model_filename)

        if args.test_data != "" and epoch > args.begin_test_epoch and epoch % args.epoch_eval == 0:
            results = eval_test(features=test_features, labels=test_labels, batch=test_batch, model=model, epoch=epoch, args=args)
            results_out = results_out.append(results)

torch.save(model.state_dict(), args.model_filename)

if args.test_data != "":
    results_out.to_csv("logs/" + args.test_data + "_logs.csv")

print("End Time : ")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
