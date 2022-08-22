from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop
from torch.nn import Linear
import torch.backends.cudnn as cudnn

from utils.utils import load_data_origin_data

from torch.utils.data import Dataset, DataLoader, random_split

import argparse, time, os, pickle
import numpy as np
from sklearn.decomposition import PCA
import scanpy as sc
from anndata import AnnData


torch.set_num_threads(2)
seed = 666

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


class MeHiSCC(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, args, v=1):
        super(MeHiSCC, self).__init__()

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

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

def plot_centre(z, center_z, ground_truth, args, xbar=None, center_full=None):
    n_cells = z.shape[0]
    n_centers = center_z.shape[0]

    ground_truth = np.array([str(i) for i in ground_truth] + ["Centers"] * n_centers)
    bulletin = np.array(["Cells"] * n_cells + ["Centers"] * n_centers)

    z_out = np.concatenate([z,center_z])

    formatting = AnnData(z_out)
    formatting.obs["cell_type"] = ground_truth
    formatting.obs["centers"] = bulletin

    sc.pp.neighbors(formatting, n_neighbors=15, use_rep='X')
    sc.tl.umap(formatting)
    sc.pl.umap(formatting, color=["cell_type", "centers"],
               save="_Centers_MeHi_SCC_" + args.name + "_Z_k_" + str(args.knn_k) +
                    "_l_" + str(args.levels) + "_t_" + str(args.tau) + "_ct_" + str(args.cent_type) + ".png")

    if xbar is not None and center_full is not None:
        xbar_out = np.concatenate([xbar, center_full])

        formatting = AnnData(xbar_out)
        formatting.obs["cell_type"] = ground_truth
        formatting.obs["centers"] = bulletin

        sc.pp.neighbors(formatting, n_neighbors=15, use_rep='X')
        sc.tl.umap(formatting)
        sc.pl.umap(formatting, color=["cell_type","centers"],
                   save="_Centers_MeHi_SCC_" + args.name + "_Xbar_k_" + str(args.knn_k) +
                       "_l_" + str(args.levels) + "_t_" + str(args.tau) + ".png")



def train_mehiscc(dataset, args):
    # Calculate Clustering Centre
    print("Loading Clustering Centre ......")
    # centre1, centre2, centre3, centre4, y_pred_last = find_cluster_centre(features=xbar.cpu().numpy(), device=args.device, args=args)
    centre_path = "centre/"
    centre_file = centre_path + 'centre_' + args.name + '_k_' + str(args.knn_k) + '_l_' + str(
        args.levels) + '_t_' + str(args.tau) + '.pkl'
    with open(centre_file, 'rb') as f:
        centre1, centre3, global_pred_labels = pickle.load(f)

    args.n_clusters = centre1.shape[0]

    model = MeHiSCC(512, 256, 64, 64, 256, 512,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                args=args,
                v=1.0).to(args.device)
    print(model)

    # Auto-Encoder Pretraining
    model.pretrain_ae(LoadDataset(dataset.x))
    data = torch.Tensor(dataset.x).to(args.device)
    y = dataset.y

    with torch.no_grad():
        xbar, _, _, _, z = model.ae(data)

    if args.cent_type == 1:
        pca = PCA(n_components=args.n_z)
        centre = pca.fit_transform(centre1)
    elif args.cent_type == 2:
        pca = PCA(n_components=args.n_z)
        centre = pca.fit_transform(centre3)
    elif args.cent_type == 3:
        centre1_tensor = torch.Tensor(centre1).to(args.device)
        _, _, _, _, centre = model.ae(centre1_tensor)
        centre = centre.detach().cpu().numpy()
    else:
        assert "--cent_type can noly be 1, 2 or 3"

    plot_centre(xbar=xbar.detach().cpu().numpy(),z=z.detach().cpu().numpy(),center_z=centre,center_full=centre1,ground_truth=y,args=args)

if __name__ == "__main__":
    print("Start Time : ")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # MeHi-SCC Settings
    parser.add_argument('--name', type=str, default='klein')
    parser.add_argument('--pre_lr', type=float, default=1e-4)

    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--load_type', type=str, default='csv')
    parser.add_argument('--similar_method', type=str, default='ncos')
    parser.add_argument('--pre_batch_size', type=int, default=32)
    parser.add_argument('--pre_epoch', type=int, default=400)
    parser.add_argument('--noise_value', type=float, default=1)
    parser.add_argument('--nlayers', type=int, default=5)
    parser.add_argument('--nhidden', type=int, default=256)
    parser.add_argument('--device', type=int, default=0)

    # Hi-Lander Settings
    parser.add_argument('--levels', type=str, default='1')
    parser.add_argument('--knn_k', type=str, default='10')
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--cent_type', type=int, default=3)

    # Device
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.device)
    print("use cuda: {}".format(args.cuda))

    args.device = torch.device("cuda" if args.cuda else "cpu")

    file_path = "data/" + args.name + ".csv"

    print(args.name)

    dataset = load_data_origin_data(file_path, args.load_type, scaling=True)
    args.k = int(len(dataset.y) / 100)

    if args.k < 5:
        args.k = 5
    if args.k > 20:
        args.k = 20

    args.n_clusters = len(np.unique(dataset.y))
    args.n_input = dataset.x.shape[1]

    train_mehiscc(dataset, args)

    print("End Time : ")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
