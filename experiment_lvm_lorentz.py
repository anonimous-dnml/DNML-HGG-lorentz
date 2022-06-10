import warnings
warnings.simplefilter('ignore')
# ParallelNativeやpytorchから要求されるwarningを無視する。
import torch
import numpy as np
import pandas as pd
import gc
import time
from copy import deepcopy
from torch.utils.data import DataLoader
from lorentz import LinkPrediction
import torch.multiprocessing as multi
from functools import partial
import os

RESULTS = "results"


def calc_metrics(device_idx, n_dim, n_nodes, n_graphs, n_devices, model_n_dims):

    for n_graph in range(n_graphs):
    # for n_graph in range(int(n_graphs * device_idx / n_devices), int(n_graphs * (device_idx + 1) / n_devices)):
        print(n_graph)

        dataset = np.load('dataset/dim_' + str(n_dim) + '/graph_' + str(n_nodes) + '_' + str(n_graph) +
                          '.npy', allow_pickle='TRUE').item()  # object型なので、itemを付けないと辞書として使えない。
        adj_mat = dataset["adj_mat"]
        params_dataset = dataset["params_adj_mat"]
        positive_samples = dataset["positive_samples"]
        negative_samples = dataset["negative_samples"]
        train_graph = dataset["train_graph"]
        lik_data = dataset["lik_data"]
        x_e = dataset["x_e"]
        # 真のローレンツ座標
        x_lorentz = np.zeros(
            (params_dataset['n_nodes'], params_dataset['n_dim'] + 1))
        for i in range(params_dataset['n_nodes']):
            x_lorentz[i, 0] = (1 + np.sum(x_e[i]**2)) / (1 - np.sum(x_e[i]**2))
            x_lorentz[i, 1:] = 2 * x_e[i] / (1 - np.sum(x_e[i]**2))

        # パラメータ
        burn_epochs = 300
        burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
        n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
        n_max_negatives = n_max_positives * 10
        lr_embeddings = 0.1
        lr_epoch_10 = 10.0 * \
            (burn_batch_size * (n_max_positives + n_max_negatives)) / \
            32 / 100  # batchサイズに対応して学習率変更
        lr_beta = 0.001
        lr_sigma = 0.001
        sigma_max = 1.0
        sigma_min = 0.001
        beta_min = 0.1
        beta_max = 10.0

        # それ以外
        loader_workers = 16
        print("loader_workers: ", loader_workers)
        shuffle = True
        sparse = False

        device = "cuda:" + str(device_idx)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 平均次数が少なくなるように手で調整する用
        print('average degree:', np.sum(adj_mat) / len(adj_mat))

        result = pd.DataFrame()
        basescore_y_and_z_list = []
        basescore_y_given_z_list = []
        basescore_z_list = []
        DNML_codelength_list = []
        pc_first_list = []
        pc_second_list = []
        AIC_naive_list = []
        BIC_naive_list = []
        CV_score_list = []
        AUC_list = []
        GT_AUC_list = []
        Cor_list = []

        for model_n_dim in model_n_dims:
            basescore_y_and_z, basescore_y_given_z, basescore_z, DNML_codelength, pc_first, pc_second, AIC_naive, BIC_naive, AUC, GT_AUC, correlation = LinkPrediction(
                adj_mat=adj_mat,
                train_graph=train_graph,
                positive_samples=positive_samples,
                negative_samples=negative_samples,
                lik_data=lik_data,
                x_lorentz=x_lorentz,
                params_dataset=params_dataset,
                model_n_dim=model_n_dim,
                burn_epochs=burn_epochs,
                burn_batch_size=burn_batch_size,
                n_max_positives=n_max_positives,
                n_max_negatives=n_max_negatives,
                lr_embeddings=lr_embeddings,
                lr_epoch_10=lr_epoch_10,
                lr_beta=lr_beta,
                lr_sigma=lr_sigma,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                beta_min=beta_min,
                beta_max=beta_max,
                device=device,
                loader_workers=16,
                shuffle=True,
                sparse=False,
                calc_groundtruth=True
            )
            basescore_y_and_z_list.append(basescore_y_and_z)
            basescore_y_given_z_list.append(basescore_y_given_z)
            basescore_z_list.append(basescore_z)
            DNML_codelength_list.append(DNML_codelength)
            pc_first_list.append(pc_first)
            pc_second_list.append(pc_second)
            AIC_naive_list.append(AIC_naive)
            BIC_naive_list.append(BIC_naive)
            AUC_list.append(AUC)
            GT_AUC_list.append(GT_AUC)
            Cor_list.append(correlation)

        result["model_n_dims"] = model_n_dims
        result["n_nodes"] = params_dataset["n_nodes"]
        result["n_dim"] = params_dataset["n_dim"]
        result["R"] = params_dataset["R"]
        result["sigma"] = params_dataset["sigma"]
        result["beta"] = params_dataset["beta"]
        result["DNML_codelength"] = DNML_codelength_list
        result["AIC_naive"] = AIC_naive_list
        result["BIC_naive"] = BIC_naive_list
        result["AUC"] = AUC_list
        result["GT_AUC"] = GT_AUC_list
        result["Cor"] = Cor_list
        result["basescore_y_and_z"] = basescore_y_and_z_list
        result["basescore_y_given_z"] = basescore_y_given_z_list
        result["basescore_z"] = basescore_z_list
        result["pc_first"] = pc_first_list
        result["pc_second"] = pc_second_list
        result["burn_epochs"] = burn_epochs
        result["burn_batch_size"] = burn_batch_size
        result["n_max_positives"] = n_max_positives
        result["n_max_negatives"] = n_max_negatives
        result["lr_embeddings"] = lr_embeddings
        result["lr_epoch_10"] = lr_epoch_10
        result["lr_beta"] = lr_beta
        result["lr_sigma"] = lr_sigma
        result["sigma_max"] = sigma_max
        result["sigma_min"] = sigma_min
        result["beta_max"] = beta_max
        result["beta_min"] = beta_min

        os.makedirs(RESULTS + "/Dim" + str(n_dim), exist_ok=True)

        result.to_csv(RESULTS + "/Dim" + str(n_dim) + "/result_" + str(n_dim) + "_" + str(n_nodes) +
                      "_" + str(n_graph) + ".csv", index=False)


if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    import argparse

    parser = argparse.ArgumentParser(description='HGG')
    parser.add_argument('n_dim', help='n_dim')
    parser.add_argument('device', help='device')
    args = parser.parse_args()
    print(args)

    n_nodes_list = [400, 800, 1600, 3200, 6400]

    if int(args.n_dim) == 8:
        model_n_dims = [2, 4, 8, 16]
    elif int(args.n_dim) == 16:
        model_n_dims = [2, 4, 8, 16, 32]

    for n_nodes in n_nodes_list:

        calc_metrics(device_idx=int(args.device), n_dim=int(args.n_dim),
                     n_nodes=n_nodes, n_graphs=10, n_devices=4, model_n_dims=model_n_dims)
