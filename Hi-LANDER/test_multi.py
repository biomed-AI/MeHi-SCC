import argparse, time, os, pickle
import pandas as pd
import numpy as np
import os
import dgl
import torch
import torch.optim as optim

from models import LANDER
from dataset import LanderDataset
from utils import evaluation, decode, build_next_level, stop_iterating

from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

###########
# ArgParser
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--test_data', type=str, required=True)
parser.add_argument('--model_path', type=str, default='checkpoint/')
parser.add_argument('--faiss_gpu', action='store_true')
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--save_path', type=str, default="results/")
parser.add_argument('--affl', type=str, default="Hi-LANDER_meta")

# HyperParam
parser.add_argument('--knn_k', type=int, default=10)
parser.add_argument('--levels', type=int, default=1)
parser.add_argument('--tau', type=float, default=0.5)
parser.add_argument('--threshold', type=str, default='prob')
parser.add_argument('--metrics', type=str, default='pairwise,bcubed,nmi,ari')
parser.add_argument('--early_stop', action='store_true')

parser.add_argument('--consider_batch', action='store_true')

# Model
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--num_conv', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--gat', action='store_true')
parser.add_argument('--gat_k', type=int, default=1)
parser.add_argument('--balance', action='store_true')
parser.add_argument('--use_cluster_feat', action='store_true')
parser.add_argument('--use_focal_loss', action='store_true')
parser.add_argument('--use_gt', action='store_true')

parser.add_argument('--cuda_no', type=str, default='0')
parser.add_argument('--cpu', action='store_true')

# Subgraph
parser.add_argument('--batch_size', type=int, default=4096)

args = parser.parse_args()
print(args)

###########################
# Environment Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_no

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if args.cpu:
    device = torch.device('cpu')

##################
# Data Preparation
if args.consider_batch:
    with open(args.data_path + args.test_data + ".pkl", 'rb') as f:
        features, labels, batch = pickle.load(f)
else:
    with open(args.data_path + args.test_data + ".pkl", 'rb') as f:
        features, labels = pickle.load(f)

global_features = features.copy()
dataset = LanderDataset(features=features, labels=labels, k=args.knn_k,
                        levels=1, faiss_gpu=args.faiss_gpu)
g = dataset.gs[0]
g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))
g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))
global_labels = labels.copy()
ids = np.arange(g.number_of_nodes())
global_edges = ([], [])
global_peaks = np.array([], dtype=np.long)
global_edges_len = len(global_edges[0])
global_num_nodes = g.number_of_nodes()

fanouts = [args.knn_k-1 for i in range(args.num_conv + 1)]
sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
# fix the number of edges
test_loader = dgl.dataloading.NodeDataLoader(
    g, torch.arange(g.number_of_nodes()), sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers
)

##################
# Model Definition
if not args.use_gt:
    feature_dim = g.ndata['features'].shape[1]
    model = LANDER(feature_dim=feature_dim, nhid=args.hidden,
                   num_conv=args.num_conv, dropout=args.dropout,
                   use_GAT=args.gat, K=args.gat_k,
                   balance=args.balance,
                   use_cluster_feat=args.use_cluster_feat,
                   use_focal_loss=args.use_focal_loss)
    model.load_state_dict(torch.load(args.model_path + args.test_data + ".ckpt"))
    model = model.to(device)
    model.eval()

# number of edges added is the indicator for early stopping
num_edges_add_last_level = np.Inf
##################################
# Predict connectivity and density
for level in range(args.levels):
    if not args.use_gt:
        total_batches = len(test_loader)
        for batch, minibatch in enumerate(test_loader):
            input_nodes, sub_g, bipartites = minibatch
            sub_g = sub_g.to(device)
            bipartites = [b.to(device) for b in bipartites]
            with torch.no_grad():
                output_bipartite = model(bipartites)
            global_nid = output_bipartite.dstdata[dgl.NID]
            global_eid = output_bipartite.edata['global_eid']
            g.ndata['pred_den'][global_nid] = output_bipartite.dstdata['pred_den'].to('cpu')
            g.edata['prob_conn'][global_eid] = output_bipartite.edata['prob_conn'].to('cpu')
            torch.cuda.empty_cache()
            if (batch + 1) % 10 == 0:
                print('Batch %d / %d for inference' % (batch, total_batches))

    new_pred_labels, peaks,\
        global_edges, global_pred_labels, global_peaks = decode(g, args.tau, args.threshold, args.use_gt,
                                                                ids, global_edges, global_num_nodes,
                                                                global_peaks)

    ids = ids[peaks]
    new_global_edges_len = len(global_edges[0])
    num_edges_add_this_level = new_global_edges_len - global_edges_len
    if stop_iterating(level, args.levels, args.early_stop, num_edges_add_this_level, num_edges_add_last_level, args.knn_k):
        break
    global_edges_len = new_global_edges_len
    num_edges_add_last_level = num_edges_add_this_level

    # build new dataset
    features, labels, cluster_features = build_next_level(features, labels, peaks,
                                                          global_features, global_pred_labels, global_peaks)
    # After the first level, the number of nodes reduce a lot. Using cpu faiss is faster.

    dataset = LanderDataset(features=features, labels=labels, k=args.knn_k,
                            levels=1, faiss_gpu=False, cluster_features=cluster_features)
    g = dataset.gs[0]

    g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))
    g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))
    test_loader = dgl.dataloading.NodeDataLoader(
        g, torch.arange(g.number_of_nodes()), sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

# print("-- Cell Type Performance --")
# evaluation(global_pred_labels, global_labels, args.metrics)
ari_ct = ARI(global_pred_labels, global_labels)
nmi_ct = NMI(global_pred_labels, global_labels)

res = {"Dataset":args.test_data, "knn_k":args.knn_k, "levels":args.levels, "tau":args.tau, "ARI_Celltype":ari_ct, "NMI_Celltype":nmi_ct}
results = pd.DataFrame(res,index=[0])
results.to_csv(args.save_path + "Result_" + args.test_data + "_" + args.affl + "_k_" + str(args.knn_k) +
               "_l_" + str(args.levels) + "_t_" + str(args.tau) + ".csv")
print("Result Saved : " + args.save_path + "Result_" + args.test_data + "_" + args.affl + "_k_" + str(args.knn_k) +
               "_l_" + str(args.levels) + "_t_" + str(args.tau) + ".csv")


args.save_cent_path = "../centre/"
centre1 = output_bipartite.ndata['features']['_N'].cpu().numpy()
centre3 = output_bipartite.dstdata['conv_features'].cpu().numpy()

f = open(args.save_cent_path + 'centre_' + args.test_data + '_k_' + str(args.knn_k) +
         '_l_' + str(args.levels) + '_t_' + str(args.tau) + '.pkl', 'wb')
pickle.dump([centre1, centre3, global_pred_labels], f)
f.close()

print("Clustering Centre Saved : " + args.save_cent_path + 'centre_' + args.test_data + '_k_' + str(args.knn_k) +
         '_l_' + str(args.levels) + '_t_' + str(args.tau) + '.pkl')
