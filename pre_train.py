from __future__ import print_function, division
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD, RMSprop
from torch.nn import Linear
import torch.backends.cudnn as cudnn
import torch.optim as optim

from utils.utils import load_data_origin_data, set_train_sampler_loader
from utils.deduce import decode

from evaluation import eva, eva_mat
from torch.utils.data import Dataset, DataLoader, random_split

import argparse, time, os, pickle
import pandas as pd
import numpy as np

import dgl

from models.lander import LANDER
from dataset.dataset import LanderDataset
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

from focal_loss import FocalLoss

from calcu_graph import construct_graph_kmean
torch.set_num_threads(2)
seed=666

cudnn.deterministic = True
cudnn.benchmark = True
import random
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        #
        self.z_layer = Linear(n_enc_3, n_z)
        #
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)


        self.x_bar_layer = Linear(n_dec_3, n_input)


    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar, enc_h1, enc_h2, enc_h3, z


class MeHi_SCC(nn.Module):

    def  __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                n_input, n_z, n_clusters, args, v=1):
        super(MeHi_SCC, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        
        self.lander = LANDER(feature_dim=n_input, nhid=n_clusters,
                             num_conv=args.num_conv, dropout=args.dropout,
                             use_GAT=args.gat, K=args.gat_k,
                             balance=args.balance,
                             use_cluster_feat=args.use_cluster_feat,
                             use_focal_loss=args.use_focal_loss)


        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

        # Lander Settings
        if args.use_focal_loss:
            self.loss_conn = FocalLoss(2)
        else:
            self.loss_conn = nn.CrossEntropyLoss()
        self.loss_den = nn.MSELoss()
        self.balance = args.balance

    def add_noise(self, inputs):
        return inputs + (torch.randn(inputs.shape) * args.noise_value)

    def pretrain_ae(self, dataset):
        train_loader = DataLoader(dataset, batch_size=args.pre_batch_size, shuffle=True)
        optimizer = Adam(self.ae.parameters(), lr=args.pre_lr)
        for epoch in range(args.pre_epoch):
            for batch_idx, (x, _) in enumerate(train_loader):
                x_noise = self.add_noise(x)
                x_noise = x_noise.cuda()
                x = x.cuda()

                x_bar, _, _, _, z = self.ae(x_noise)
                loss = F.mse_loss(x_bar, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def lander_loss(self, bipartite):

        pred_den = bipartite.dstdata['pred_den']
        loss_den = self.loss_den(pred_den, bipartite.dstdata['density'])

        labels_conn = bipartite.edata['labels_conn']
        mask_conn = bipartite.edata['mask_conn']

        if self.balance:
            labels_conn = bipartite.edata['labels_conn']
            neg_check = torch.logical_and(bipartite.edata['labels_conn'] == 0, mask_conn)
            num_neg = torch.sum(neg_check).item()
            neg_indices = torch.where(neg_check)[0]
            pos_check = torch.logical_and(bipartite.edata['labels_conn'] == 1, mask_conn)
            num_pos = torch.sum(pos_check).item()
            pos_indices = torch.where(pos_check)[0]
            if num_pos > num_neg:
                mask_conn[pos_indices[np.random.choice(num_pos, num_pos - num_neg, replace=False)]] = 0
            elif num_pos < num_neg:
                mask_conn[neg_indices[np.random.choice(num_neg, num_neg - num_pos, replace=False)]] = 0

        # In subgraph training, it may happen that all edges are masked in a batch
        if mask_conn.sum() > 0:
            loss_conn = self.loss_conn(bipartite.edata['pred_conn'][mask_conn], labels_conn[mask_conn])
            loss = loss_den + loss_conn
            loss_den_val = loss_den.item()
            loss_conn_val = loss_conn.item()
        else:
            loss = loss_den
            loss_den_val = loss_den.item()
            loss_conn_val = 0

        return loss, loss_den_val, loss_conn_val


    def forward(self, x):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)


        #Lander Prediction
        global_features = x
        kmeans = KMeans(n_clusters=args.k, random_state=1).fit(z.detach().cpu().numpy())
        labels = kmeans.labels_

        dataset_lander = LanderDataset(features=global_features.detach().cpu().numpy(), labels=labels, k=args.k,
                                levels=1, faiss_gpu=args.faiss_gpu)
        g = dataset_lander.gs[0]
        g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))
        g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))
        global_labels = labels.copy()
        ids = np.arange(g.number_of_nodes())
        global_edges = ([], [])
        global_peaks = np.array([], dtype=np.long)
        global_edges_len = len(global_edges[0])
        global_num_nodes = g.number_of_nodes()

        fanouts = [args.k - 1 for i in range(args.num_conv + 1)] ###
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
        # fix the number of edges
        test_loader = dgl.dataloading.NodeDataLoader(
            g, torch.arange(g.number_of_nodes()), sampler,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers
        )

        num_edges_add_last_level = np.Inf
        if not args.use_gt:
            total_batches = len(test_loader)
            for batch, minibatch in enumerate(test_loader):
                input_nodes, sub_g, bipartites = minibatch
                sub_g = sub_g.to(args.device)
                bipartites = [b.to(args.device) for b in bipartites]
                with torch.no_grad():
                    output_bipartite = self.lander(bipartites)
                global_nid = output_bipartite.dstdata[dgl.NID]
                global_eid = output_bipartite.edata['global_eid']
                g.ndata['pred_den'][global_nid] = output_bipartite.dstdata['pred_den'].to('cpu')
                g.edata['prob_conn'][global_eid] = output_bipartite.edata['prob_conn'].to('cpu')
                torch.cuda.empty_cache()
                if (batch + 1) % 10 == 0:
                    print('Batch %d / %d for inference' % (batch, total_batches))

        # predict = self.gcn_model(x, adj)
        predict_bipartites = output_bipartite
        predict = predict_bipartites.dstdata['conv_features']
        _, _, _, lan_label, _ = decode(g, args.tau, args.threshold, args.use_gt,
                                                                ids, global_edges, global_num_nodes,
                                                                global_peaks)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, lan_label


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def train_mehi_scc(dataset, args):

    model = MeHi_SCC(512, 256, 64, 64, 256, 512,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                args=args,
                v=1.0).to(args.device)

    model = model.to(args.device)

    print(model.eval())

    # Train - Test Assignment
    args.test_ratio = 1 - args.train_ratio

    print(args.name)
    features = dataset.x

    cell_num = features.shape[0]

    # Auto-Encoder Pretraining
    model.pretrain_ae(LoadDataset(dataset.x))

    data = torch.Tensor(dataset.x).to(args.device)

    with torch.no_grad():
        xbar, _, _, _, z = model.ae(data)

    print("##### Performance of Train Pre-training #####")

    train_ground_truth = dataset.y
    train_batch = dataset.b

    print("Origin Profile before AE : ")
    eva_mat(dataset.x, ct_lbs=train_ground_truth, bc_lbs=train_batch)
    print("X bar from AE : ")
    eva_mat(xbar.detach().cpu().numpy(), ct_lbs=train_ground_truth, bc_lbs=train_batch)
    print("Latent Z bar from AE : ")
    eva_mat(z.detach().cpu().numpy(), ct_lbs=train_ground_truth, bc_lbs=train_batch)

    features = z.detach().cpu().numpy()
    labels = dataset.y
    batch = dataset.b

    f = open('Hi-LANDER/data/' + args.name + '_z.pkl', 'wb')
    pickle.dump([features, labels, batch], f)
    f.close()

    features = xbar.detach().cpu().numpy()
    labels = dataset.y
    batch = dataset.b

    f = open('Hi-LANDER/data/' + args.name + '_bar.pkl', 'wb')
    pickle.dump([features, labels, batch], f)
    f.close()

if __name__ == "__main__":
    print("Start Time : ")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # MeHi-SCC Settings
    parser.add_argument('--name', type=str, default='klein')
    parser.add_argument('--pre_lr', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.01)


    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--load_type', type=str, default='csv')
    parser.add_argument('--kl_loss', type=float, default=0.1)
    parser.add_argument('--ce_loss', type=float, default=0.01)
    parser.add_argument('--re_loss', type=float, default=1)
    parser.add_argument('--similar_method', type=str, default='ncos')
    parser.add_argument('--pre_batch_size', type=int, default=32)
    parser.add_argument('--pre_epoch', type=int, default=400)
    parser.add_argument('--noise_value', type=float, default=1)
    parser.add_argument('--nlayers', type=int, default=5)
    parser.add_argument('--nhidden', type=int, default=256)
    parser.add_argument('--device', type=int, default=0)

    # Hi-Lander Settings
    # Dataset
    parser.add_argument('--levels', type=str, default='1')
    parser.add_argument('--faiss_gpu', action='store_true')
    parser.add_argument('--model_filename', type=str, default='lander.pth')

    # KNN
    parser.add_argument('--knn_k', type=str, default='10')
    parser.add_argument('--num_workers', type=int, default=0)

    # Lander Model
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--num_conv', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--gat', action='store_true')
    parser.add_argument('--gat_k', type=int, default=1)
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--use_cluster_feat', action='store_true')
    parser.add_argument('--use_focal_loss', action='store_true')
    parser.add_argument('--use_gt', action='store_true')
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--threshold', type=str, default='prob')

    # Device
    parser.add_argument('--cpu', action='store_true')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--den_loss', type=float, default=1)
    parser.add_argument('--conn_loss', type=float, default=1)

    # Optimizer
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # Training - Test Ratio
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of Training Set')
    parser.add_argument('--seed', type=int, default=100, help='Random Seed')


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.device)
    print("use cuda: {}".format(args.cuda))

    args.device = torch.device("cuda" if args.cuda else "cpu")

    file_path = "data/" + args.name +".csv"
             
    print(args.name)

    dataset = load_data_origin_data(file_path, args.load_type,scaling=True)
    args.k = int(len(dataset.y)/100)

    if args.k < 5:
        args.k = 5
    if args.k > 20:
        args.k = 20

    args.n_clusters = len(np.unique(dataset.y))
    args.n_input = dataset.x.shape[1]

    train_mehi_scc(dataset, args)

    print("End Time : ")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
