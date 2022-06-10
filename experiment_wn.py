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
from lorentz import CV_HGG, DNML_HGG, LinkPrediction, create_test_for_link_prediction
from lorentz import arcosh, h_dist
import torch.multiprocessing as multi
from functools import partial
from scipy.io import mmread
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import os

RESULTS = "results"


def create_wn_dataset(dataset_name):

    df = pd.read_csv("dataset/others/" + dataset_name + "_closure.csv")

    node_names = set(df["id1"]) | set(df["id2"])
    node_names = np.array(list(node_names))

    n_nodes = len(node_names)

    adj_mat = np.zeros((n_nodes, n_nodes))
    is_a = np.zeros((len(df), 2))

    for index, r in df.iterrows():
        u = np.where(node_names == r[0])[0]
        v = np.where(node_names == r[1])[0]
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1

        is_a[index, 0] = u
        is_a[index, 1] = v

    adj_mat = adj_mat.astype(np.int)
    is_a = is_a.astype(np.int)

    params_dataset = {
        "n_nodes": n_nodes
    }

    print(node_names)
    print(adj_mat)
    print(np.sum(adj_mat))
    print(is_a)

    os.makedirs(RESULTS+"/" + dataset_name, exist_ok=True)

    np.save(RESULTS + "/" + dataset_name + "/adj_mat.npy", adj_mat)
    np.save(RESULTS + "/" + dataset_name + "/is_a.npy", is_a)


def is_a_score(is_a, n_dim, lorentz_table, alpha=1000):

    score_sum = 0
    for r in is_a:
        u = r[0]
        v = r[1]
        c_u = torch.Tensor(lorentz_table[u])
        c_v = torch.Tensor(lorentz_table[v])
        r_u = arcosh(c_u[0])
        r_v = arcosh(c_v[0])
        dst = h_dist(c_u.reshape((1, -1)), c_v.reshape((1, -1)))
        score = -(1 + alpha * (r_v - r_u)) * dst
        # print(score)
        score_sum += score[0]
        pass

    print("Dim ", n_dim, ": ", score_sum / len(is_a))

    return score_sum / len(is_a)


def calc_metrics_realworld(device_idx, model_n_dims, dataset_name):

    create_wn_dataset(dataset_name)

    adj_mat = np.load(RESULTS + "/" + dataset_name + "/adj_mat.npy")
    is_a = np.load(RESULTS + "/" + dataset_name + "/is_a.npy")

    print("n_nodes:", len(adj_mat))
    print("n_edges:", np.sum(adj_mat) / 2)
    n_nodes = len(adj_mat)

    params_dataset = {
        'n_nodes': n_nodes,
        'R': np.log(n_nodes) - 0.5,
    }

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
    is_a_score_list = []

    for model_n_dim in model_n_dims:
        basescore_y_and_z, basescore_y_given_z, basescore_z, DNML_codelength, pc_first, pc_second, AIC_naive, BIC_naive, lorentz_table = DNML_HGG(
            adj_mat=adj_mat,
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
        )
        basescore_y_and_z_list.append(basescore_y_and_z)
        basescore_y_given_z_list.append(basescore_y_given_z)
        basescore_z_list.append(basescore_z)
        DNML_codelength_list.append(DNML_codelength)
        pc_first_list.append(pc_first)
        pc_second_list.append(pc_second)
        AIC_naive_list.append(AIC_naive)
        BIC_naive_list.append(BIC_naive)

        is_a_score_list.append(is_a_score(is_a, model_n_dim, lorentz_table))
        np.save(RESULTS + "/" + dataset_name + "/embedding_" +
                str(model_n_dim) + ".npy", lorentz_table)

    result["model_n_dims"] = model_n_dims
    result["n_nodes"] = params_dataset["n_nodes"]
    # result["n_dim"] = params_dataset["n_dim"]
    result["R"] = params_dataset["R"]
    # result["sigma"] = params_dataset["sigma"]
    # result["beta"] = params_dataset["beta"]
    result["DNML_codelength"] = DNML_codelength_list
    result["AIC_naive"] = AIC_naive_list
    result["BIC_naive"] = BIC_naive_list
    # result["AUC"] = AUC_list
    # result["GT_AUC"] = GT_AUC_list
    # result["Cor"] = Cor_list
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

    result.to_csv(RESULTS + "/" + dataset_name + "/result.csv", index=False)

def result_wn(model_n_dims, dataset_name):
    is_a_score_list = []

    for model_n_dim in model_n_dims:
        lorentz_table = np.load(
            RESULTS + '/' + dataset_name + '/embedding_' + str(model_n_dim) + '.npy')
        is_a = np.load('results/' + dataset_name + '/is_a.npy')
        is_a_score_list.append(is_a_score(
            is_a, model_n_dim, lorentz_table, 100))

    result = pd.read_csv(RESULTS + "/" + dataset_name + "/result.csv")

    result_MinGE = pd.read_csv(
        RESULTS + "/" + dataset_name + "/result_MinGE.csv")

    result = pd.merge(result, result_MinGE, on="model_n_dims", how="left")

    D_DNML = result["model_n_dims"].values[
        np.argmin(result["DNML_codelength"].values)]
    D_AIC = result["model_n_dims"].values[
        np.argmin(result["AIC_naive"].values)]
    D_BIC = result["model_n_dims"].values[
        np.argmin(result["BIC_naive"].values)]
    D_MinGE = result["model_n_dims"].values[
        np.argmin(result["MinGE"].values)]

    print("DNML:", D_DNML)
    print("AIC:", D_AIC)
    print("BIC:", D_BIC)
    print("MinGE:", D_MinGE)

    plt.clf()
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    def normalize(x):
        return (x - np.min(x.values)) / (np.max(x.values) - np.min(x.values))

    result["DNML_codelength"] = normalize(result["DNML_codelength"])
    result["AIC_naive"] = normalize(result["AIC_naive"])
    result["BIC_naive"] = normalize(result["BIC_naive"])
    result["MinGE"] = normalize(result["MinGE"])

    ax.plot(result["model_n_dims"], result[
            "DNML_codelength"], label="DNML-HGG", color="red")
    ax.plot(result["model_n_dims"], result["AIC_naive"],
            label="AIC_naive", color="blue")
    ax.plot(result["model_n_dims"], result["BIC_naive"],
            label="BIC_naive", color="green")
    ax.plot(result["model_n_dims"], result[
            "MinGE"], label="MinGE", color="orange")
    plt.xscale('log')
    plt.xticks(result["model_n_dims"], fontsize=20)
    plt.yticks(fontsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=15)
    ax.set_xlabel("Dimensionality", fontsize=20)
    ax.set_ylabel("Normalized Criterion", fontsize=20)
    plt.tight_layout()

    plt.savefig(RESULTS + "/" + dataset_name +
                "/result_" + dataset_name + ".png")


if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    import argparse

    parser = argparse.ArgumentParser(description='HGG')
    parser.add_argument('dataset_name', help='dataset_name')
    parser.add_argument('device', help='device')

    args = parser.parse_args()
    print(args)

    model_n_dims = [2, 4, 8, 16, 32, 64]

    calc_metrics_realworld(device_idx=int(
        args.device), model_n_dims=model_n_dims, dataset_name=args.dataset_name)


