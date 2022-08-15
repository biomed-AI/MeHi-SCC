import pandas as pd
import numpy as np
import dgl
import torch
import pickle
from dataset import LanderDataset
from utils import decode, build_next_level, stop_iterating

from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

def eval_test(features, labels, model, epoch, args, batch=None):
    test_k_list = [int(k) for k in args.test_knn_k.split(',')]
    test_lvl_list = [int(l) for l in args.test_levels.split(',')]
    print("KNN_k : ")
    print(test_k_list)
    print("Levels : ")
    print(test_lvl_list)

    if batch is not None and len(np.unique(batch)) != 1:
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


    for test_knn_k, test_levels in zip(test_k_list, test_lvl_list):
        print("Testing on k = " + str(test_knn_k))
        print("Testing on l = " + str(test_levels))

        global_features = features.copy()
        dataset = LanderDataset(features=features, labels=labels, k=test_knn_k,
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

        fanouts = [test_knn_k - 1 for i in range(args.num_conv + 1)]
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
        # fix the number of edges
        test_loader = dgl.dataloading.NodeDataLoader(
            g, torch.arange(g.number_of_nodes()), sampler,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers
        )

        # number of edges added is the indicator for early stopping
        num_edges_add_last_level = np.Inf
        ##################################
        # Predict connectivity and density
        for level in range(test_levels):
            if not args.use_gt:
                total_batches = len(test_loader)
                for batch, minibatch in enumerate(test_loader):
                    input_nodes, sub_g, bipartites = minibatch
                    sub_g = sub_g.to(args.device)
                    bipartites = [b.to(args.device) for b in bipartites]
                    with torch.no_grad():
                        output_bipartite = model(bipartites)
                    global_nid = output_bipartite.dstdata[dgl.NID]
                    global_eid = output_bipartite.edata['global_eid']
                    g.ndata['pred_den'][global_nid] = output_bipartite.dstdata['pred_den'].to('cpu')
                    g.edata['prob_conn'][global_eid] = output_bipartite.edata['prob_conn'].to('cpu')
                    torch.cuda.empty_cache()
                    if (batch + 1) % 10 == 0:
                        print('Batch %d / %d for inference' % (batch, total_batches))

            new_pred_labels, peaks, \
            global_edges, global_pred_labels, global_peaks = decode(g, args.tau, args.threshold, args.use_gt,
                                                                    ids, global_edges, global_num_nodes,
                                                                    global_peaks)

            ids = ids[peaks]
            new_global_edges_len = len(global_edges[0])
            num_edges_add_this_level = new_global_edges_len - global_edges_len
            if stop_iterating(level, test_levels, args.early_stop, num_edges_add_this_level, num_edges_add_last_level,
                              test_knn_k):
                break
            global_edges_len = new_global_edges_len
            num_edges_add_last_level = num_edges_add_this_level

            # build new dataset
            features, labels, cluster_features = build_next_level(features, labels, peaks,
                                                                  global_features, global_pred_labels, global_peaks)
            # After the first level, the number of nodes reduce a lot. Using cpu faiss is faster.

            dataset = LanderDataset(features=features, labels=labels, k=test_knn_k,
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

        # f = open('temp_perf.pkl', 'wb')
        # pickle.dump([global_pred_labels, global_labels], f)
        # f.close()
        #
        # with open("temp_perf.pkl", 'rb') as f:
        #     global_pred_labels, global_labels = pickle.load(f)
        # f.close()

        ari_ct = ARI(global_pred_labels, global_labels)
        nmi_ct = NMI(global_pred_labels, global_labels)

        if batch is not None and len(np.unique(batch)) != 1:
            ari_bc = ARI(global_pred_labels, batch)
            nmi_bc = NMI(global_pred_labels, batch)
            print("Test Dataset : " + args.test_data + " , Epoch : {} , CellType ARI : {} , CellType NMI : {} "
                                                       ", Batch ARI : {} , Batch NMI : {}".format(epoch, ari_ct, nmi_ct, ari_bc, nmi_bc))
            results = pd.DataFrame([{"Dataset": args.test_data, "Epoch": epoch,
                                     "KNN_K":test_knn_k, "Levels": test_levels,
                                     "ARI_CT": ari_ct, "NMI_CT": nmi_ct,
                                     "ARI_BC": ari_bc, "NMI_BC": nmi_bc}])
        else:
            print("Test Dataset : " + args.test_data + " , Epoch : {} , CellType ARI : {} , CellType NMI : {} ".format(epoch, ari_ct, nmi_ct))
            results = pd.DataFrame([{"Dataset": args.test_data, "Epoch": epoch,
                                     "KNN_K":test_knn_k, "Levels": test_levels,
                                     "ARI_CT": ari_ct, "NMI_CT": nmi_ct}])

        results_out = results_out.append(results)

    return results_out