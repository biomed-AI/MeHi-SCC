from __future__ import print_function, division

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD, RMSprop
from torch.nn import Linear
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, random_split

from calcu_graph import construct_graph_kmean
from utils.utils import load_data_origin_data, load_graph, cal_center
# from find_cluster_centre import find_cluster_centre
from show_centre import plot_centre
from GNN import GCNII
from evaluation import eva, eva_mat
from plot_umap import plot_umap

import argparse, time, os, pickle
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData

from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.decomposition import PCA

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

        self.gcn_model = GCNII(nfeat=n_input,
                               nlayers=args.nlayers,
                               nhidden=args.nhidden,
                               nclass=n_clusters,
                               dropout=0,
                               lamda=0.3,
                               alpha=0.2,
                               variant=False)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def add_noise(self, inputs):
        return inputs + (torch.randn(inputs.shape) * args.noise_value)

    def cluster_centre_init(self, n_clusters, n_z):
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

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

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        zg = self.gcn_model(x, adj)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, zg, z


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
    # Calculate Clustering Centre
    print("Loading Clustering Centre ......")
    # centre1, centre2, centre3, centre4, y_pred_last = find_cluster_centre(features=xbar.cpu().numpy(), device=args.device, args=args)
    centre_path = "centre/"
    centre_file = centre_path + 'centre_' + args.name + '_bar_k_' + str(args.knn_k) + '_l_' + str(
        args.levels) + '_t_' + str(args.tau) + '.pkl'
    with open(centre_file, 'rb') as f:
        centre1, centre3, global_pred_labels = pickle.load(f)

    if args.cent_type == 1:
        pca = PCA(n_components=args.n_z)
        centre = pca.fit_transform(centre1)
        args.n_clusters = centre.shape[0]
    elif args.cent_type == 2:
        pca = PCA(n_components=args.n_z)
        centre = pca.fit_transform(centre3)
        args.n_clusters = centre.shape[0]
    elif args.cent_type == 3:
        args.n_clusters = centre1.shape[0]
    elif args.cent_type == 4:
        # z_np = z.detach().cpu().numpy()
        # centre = cal_center(features=z_np,labels=global_pred_labels)
        args.n_clusters = len(np.unique(global_pred_labels))
    else:
        assert "--cent_type can noly be 1, 2, 3 or 4"


    #args.n_clusters = len(np.unique(global_pred_labels))

    model = MeHi_SCC(512, 256, 64, 64, 256, 512,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                args=args,
                v=1.0).to(args.device)
    print(model)

    # Auto-Encoder Pretraining
    model.pretrain_ae(LoadDataset(dataset.x))
    optimizer = Adam(model.parameters(), lr=args.lr)
    data = torch.Tensor(dataset.x).to(args.device)
    y = dataset.y

    with torch.no_grad():
        xbar, _, _, _, z = model.ae(data)

    y_pred_last = global_pred_labels

    ari_ct_lander = ARI(y_pred_last, dataset.y)
    nmi_ct_lander = NMI(y_pred_last, dataset.y)
    print("Done!")

    if args.cent_type == 3:
        centre1_tensor = torch.Tensor(centre1).to(args.device)
        _, _, _, _, centre = model.ae(centre1_tensor)
        centre = centre.detach().cpu().numpy()
    elif args.cent_type == 4:
        z_np = z.detach().cpu().numpy()
        centre = cal_center(features=z_np,labels=global_pred_labels)

    model.cluster_layer.data = torch.tensor(centre).to(args.device)

    features = z.data.cpu().numpy()
    error_rate = construct_graph_kmean(args.name, features.copy(), y, y,
                                       load_type='csv', topk=args.k, method='ncos')
    adj = load_graph(args.name, k=args.k, n=dataset.x.shape[0])
    adj = adj.cuda()

    patient = 0
    series = False
    pca = PCA(n_components=args.n_clusters)

    print("Start Training ......")
    for epoch in range(args.det_epoch + args.train_epoch):
        if epoch < args.det_epoch :
            cal_det = True
        else:
            cal_det = False
        if epoch % 1 == 0:
        # update_interval
            xbar, tmp_q, zg, z = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            if args.GCNII:
                pred = zg
            else:
                pred_pca = pca.fit_transform(xbar.data.cpu().numpy())
                pred = F.softmax(torch.tensor(pred_pca).to(args.device), dim=1)

            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            Q_acc, Q_nmi, Q_ari = eva(y, res1, str(epoch) + 'Q', pp=False)
            Z_acc, Z_nmi, Z_ari = eva(y, res2, str(epoch) + 'Z', pp=False)
            P_acc, P_nmi, p_ari = eva(y, res3, str(epoch) + 'P', pp=False)
            print(epoch, ':Q_acc {:.5f}'.format(Q_acc), ', Q_nmi {:.5f}'.format(Q_nmi), ', Q_ari {:.5f}'.format(Q_ari))
            print(epoch, ':Z_acc {:.5f}'.format(Z_acc), ', Z_nmi {:.5f}'.format(Z_nmi), ', Z_ari {:.5f}'.format(Z_ari))
            print(epoch, ':P_acc {:.5f}'.format(P_acc), ', P_nmi {:.5f}'.format(P_nmi), ', p_ari {:.5f}'.format(p_ari))

            delta_label = np.sum(res2 != y_pred_last).astype(np.float32) / res2.shape[0]
            y_pred_last = res2
            if epoch > 0 and delta_label < 0.0001:
                if series:
                    patient+=1
                else:
                    patient = 0
                series=True
                if patient==300:
                   print('Reached tolerance threshold. Stopping training.')
                   print("Z_acc: {}".format(Z_acc), "Z_nmi: {}".format(Z_nmi),
                            "Z_ari: {}".format(Z_ari))

                   break
            else:
                series=False

        x_bar, q, zg, z = model(data, adj)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        if cal_det:
            centres_fea = model.cluster_layer.data
            det_loss = torch.norm(centres_fea @ centres_fea.T)
            loss = args.kl_loss * kl_loss + args.ce_loss * ce_loss + re_loss - args.det_loss * det_loss
            print(epoch, ':kl_loss {:.5f}'.format(kl_loss), ', ce_loss {:.5f}'.format(ce_loss),
                  ', re_loss {:.5f}'.format(re_loss), ', -det_loss {:.5f}'.format(-det_loss), ', total_loss {:.5f}'.format(loss))
        else:
            loss = args.kl_loss * kl_loss + args.ce_loss * ce_loss + re_loss
            print(epoch, ':kl_loss {:.5f}'.format(kl_loss), ', ce_loss {:.5f}'.format(ce_loss),
                  ', re_loss {:.5f}'.format(re_loss), ', total_loss {:.5f}'.format(loss))

        if epoch == args.det_epoch:
            centres_fea = model.cluster_layer.data
            _, _, _, _, z_show = model.ae(data)
            plot_centre(z=z_show.detach().cpu().numpy(), center_z=centres_fea.detach().cpu().numpy(), ground_truth=y, args=args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Final P-Q : ")
    Q_acc, Q_nmi, Q_ari = eva(y, res1, str(epoch) + 'Q', pp=False)
    Z_acc, Z_nmi, Z_ari = eva(y, res2, str(epoch) + 'Z', pp=False)
    P_acc, P_nmi, P_ari = eva(y, res3, str(epoch) + 'P', pp=False)
    print(epoch, ':Q_acc {:.4f}'.format(Q_acc), ', Q_nmi {:.4f}'.format(Q_nmi), ', Q_ari {:.4f}'.format(Q_ari))
    print(epoch, ':Z_acc {:.4f}'.format(Z_acc), ', Z_nmi {:.4f}'.format(Z_nmi), ', Z_ari {:.4f}'.format(Z_ari))
    print(epoch, ':P_acc {:.4f}'.format(P_acc), ', P_nmi {:.4f}'.format(P_nmi), ', P_ari {:.4f}'.format(P_ari))
    print(args)

    print("##### Final Performance #####")
    xbar, tmp_q, zg, z = model(data, adj)
    ground_truth = dataset.y
    batch = dataset.b

    print("X bar from AE : ")
    ari_ct_xbar, nmi_ct_xbar, _, _ = eva_mat(xbar.detach().cpu().numpy(), ct_lbs=ground_truth, bc_lbs=batch)
    print("Latent Z bar from AE : ")
    ari_ct_z, nmi_ct_z, _, _ = eva_mat(z.detach().cpu().numpy(), ct_lbs=ground_truth, bc_lbs=batch)

    #Record UMAP
    plot_umap(xbar=xbar.detach().cpu().numpy(),z=z.detach().cpu().numpy(),
              ground_truth=ground_truth,res1=res1,res2=res2,res3=res3,
              lander_labels=y_pred_last,args=args)

    #Record Results
    n_clu_LANDER = len(np.unique(global_pred_labels))
    n_clu_Q = len(np.unique(res1))
    n_clu_Z = len(np.unique(res2))
    n_clu_P = len(np.unique(res3))

    res = {"Dataset": args.name, "N_clusters": args.n_clusters, "CT": args.cent_type,
           "knn_k": args.knn_k, "levels": args.levels, "tau": args.tau,
           "n_clu_LANDER": n_clu_LANDER,"n_clu_Q": n_clu_Q,"n_clu_Z": n_clu_Z,"n_clu_P": n_clu_P,
           "ARI_Celltype_Xbar": ari_ct_xbar, "NMI_Celltype_Xbar": nmi_ct_xbar,
           "ARI_Celltype_Z": ari_ct_z, "NMI_Celltype_Z": nmi_ct_z,
           "ARI_Celltype_Lander": ari_ct_lander, "NMI_Celltype_Lander": nmi_ct_lander,
           "Q_acc": Q_acc, "Q_nmi": Q_nmi, "Q_ari": Q_ari,
           "Z_acc": Z_acc, "Z_nmi": Z_nmi, "Z_ari": Z_ari,
           "P_acc": P_acc, "P_nmi": P_nmi, "p_ari": P_ari}
    results = pd.DataFrame(res, index=[0])
    results.to_csv("results/Result_" + args.name + "_MeHi-SCC_ct_" + str(args.cent_type) + "_k_" + str(args.knn_k) +
                   "_l_" + str(args.levels) + "_t_" + str(args.tau) + ".csv")
    print("Result Saved : " + "results/Result_" + args.name + "_MeHi-SCC_ct_" + str(args.cent_type) + "_k_" + str(args.knn_k) +
          "_l_" + str(args.levels) + "_t_" + str(args.tau) + ".csv")

    emb = z.detach().cpu().numpy()
    formatting = AnnData(emb)
    sc.pp.neighbors(formatting, n_neighbors=15, use_rep='X')
    sc.tl.umap(formatting)

    umap_df = pd.DataFrame({"UMAP_1":formatting.obsm['X_umap'][:,0],"UMAP_2":formatting.obsm['X_umap'][:,1]})
    umap_df["Lander_clu"] = global_pred_labels
    umap_df["Res1_clu"] = res1
    umap_df["Res2_clu"] = res2
    umap_df["Res3_clu"] = res3
    umap_df["knn_k"] = args.knn_k
    umap_df["levels"] = args.levels
    umap_df["tau"] = args.tau
    umap_df["Data"] = args.name
    umap_df["Method"] = "Ours"
    umap_df.to_csv("results/UMAP&Clu_" + args.name + "_MeHi-SCC_ct_" + str(args.cent_type) + "_k_" + str(args.knn_k) +
                   "_l_" + str(args.levels) + "_t_" + str(args.tau) + ".csv")
    print("Result Saved : " + "results/UMAP&Clu_" + args.name + "_MeHi-SCC_ct_" + str(args.cent_type) + "_k_" + str(args.knn_k) +
          "_l_" + str(args.levels) + "_t_" + str(args.tau) + ".csv")


if __name__ == "__main__":
    print("Start Time : ")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # MeHi-SCC Settings
    parser.add_argument('--name', type=str, default='klein')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--pre_lr', type=float, default=1e-4)
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--load_type', type=str, default='csv')
    parser.add_argument('--kl_loss', type=float, default=0.1)
    parser.add_argument('--ce_loss', type=float, default=0.01)
    parser.add_argument('--re_loss', type=float, default=1)
    parser.add_argument('--det_loss', type=float, default=1)
    parser.add_argument('--similar_method', type=str, default='ncos')
    parser.add_argument('--pre_batch_size', type=int, default=32)
    parser.add_argument('--pre_epoch', type=int, default=400)
    parser.add_argument('--train_epoch', type=int, default=1000)
    parser.add_argument('--det_epoch', type=int, default=200)
    parser.add_argument('--noise_value', type=float, default=1)
    parser.add_argument('--nlayers', type=int, default=5)
    parser.add_argument('--nhidden', type=int, default=256)

    parser.add_argument('--GCNII', action='store_true')
    parser.add_argument('--cent_type', type=int, default=4)

    # Hi-Lander Settings
    # Dataset
    parser.add_argument('--model_path', type=str, default='checkpoint/')
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--faiss_gpu', action='store_true')
    parser.add_argument('--model_filename', type=str, default='lander.pth')

    # HyperParam
    parser.add_argument('--knn_k', type=int, default=10)
    parser.add_argument('--levels', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--threshold', type=str, default='prob')
    parser.add_argument('--early_stop', action='store_true')

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

    # Subgraph
    parser.add_argument('--batch_size', type=int, default=4096)

    # Device
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--cuda_no', type=str, default='0')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.device)
    print("use cuda: {}".format(args.cuda))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_no
    import torch
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())
    args.device = torch.device("cuda" if args.cuda else "cpu")

    if args.cpu:
        args.device = torch.device("cpu")

    file_path = "data/" + args.name + ".csv"
             
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
