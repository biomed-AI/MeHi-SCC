import argparse
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

###########
# ArgParser
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--res_path', type=str, default="/home/panzx/Hi-LANDER/results/")
parser.add_argument('--test_data', type=str, default="baron_human_bar,baron_mouse_bar,biase_bar,darmanis_bar,deng_bar,goolam_bar,klein_bar,li_bar,pbmc_68k_bar,romanov_bar,segerstolpe_bar,shekhar_mouse_retina_bar,tasic_bar,xin_bar,zeisel_bar")
parser.add_argument('--save_path', type=str, default="/home/panzx/Hi-LANDER/results/aggr/")
parser.add_argument('--affl', type=str, default="Hi-LANDER_meta")

# HyperParam
parser.add_argument('--knn_k', type=str, default="3,4,5,6,7,8,9,10")
parser.add_argument('--levels', type=str, default="1,2,3,4,5")
parser.add_argument('--tau', type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")

args = parser.parse_args()

dat_list = [str(d) for d in args.test_data.split(',')]
k_list = [str(k) for k in args.knn_k.split(',')]
lvl_list = [str(l) for l in args.levels.split(',')]
tau_list = [str(t) for t in args.tau.split(',')]

temp = {"Dataset":"", "knn_k":0, "levels":0, "tau":0.0, "ARI_Celltype":0.0000, "NMI_Celltype":0.0000}
res_out = pd.DataFrame(temp,index=[0]).drop(index=0)

for i_dat in range(len(dat_list)):
    for i_k in range(len(k_list)):
        for i_l in range(len(lvl_list)):
            for i_t in range(len(tau_list)):
                dat = dat_list[i_dat]
                k = k_list[i_k]
                l = lvl_list[i_l]
                t = tau_list[i_t]

                file_name = "Result_" + dat + "_" + args.affl + "_k_" + k + \
                            "_l_" + l + "_t_" + t + ".csv"
                file_dir = args.res_path + file_name
                if os.path.exists(file_dir):
                    res = pd.read_csv(file_dir,index_col=0)
                    res_out = res_out.append(res)

res_out.to_csv(args.save_path + "Results_" + args.affl + ".csv")
print("Result Saved : " + args.save_path + "Results_" + args.affl + ".csv")
