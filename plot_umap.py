import scanpy as sc
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
import os
dirname = os.getcwd()
print(dirname)
from anndata import AnnData
import numpy as np

def plot_umap(xbar, z, ground_truth, res1, res2, res3, lander_labels, args):
    ground_truth = np.array([str(i) for i in ground_truth])
    res1 = np.array([str(i) for i in res1])
    res2 = np.array([str(i) for i in res2])
    res3 = np.array([str(i) for i in res3])
    lander_labels = np.array([str(i) for i in lander_labels])

    formatting = AnnData(xbar)
    formatting.obs["cell_type"] = ground_truth
    formatting.obs["res1"] = res1
    formatting.obs["res2"] = res2
    formatting.obs["res3"] = res3
    formatting.obs["lander_labels"] = lander_labels

    sc.pp.neighbors(formatting, n_neighbors=15, use_rep='X')
    sc.tl.umap(formatting)
    sc.pl.umap(formatting, color=["cell_type","res1","res2","res3","lander_labels"],
               save="_MeHi_SCC_" + args.name + "_Xbar_ct_" + str(args.cent_type) + "_k_" + str(args.knn_k) +
                   "_l_" + str(args.levels) + "_t_" + str(args.tau) + ".png")

    formatting = AnnData(z)
    formatting.obs["cell_type"] = ground_truth
    formatting.obs["res1"] = res1
    formatting.obs["res2"] = res2
    formatting.obs["res3"] = res3
    formatting.obs["lander_labels"] = lander_labels

    sc.pp.neighbors(formatting, n_neighbors=15, use_rep='X')
    sc.tl.umap(formatting)
    sc.pl.umap(formatting, color=["cell_type","res1","res2","res3","lander_labels"],
               save="_MeHi_SCC_" + args.name + "_Z_ct_" + str(args.cent_type) + "_k_" + str(args.knn_k) +
                    "_l_" + str(args.levels) + "_t_" + str(args.tau) + ".png")
