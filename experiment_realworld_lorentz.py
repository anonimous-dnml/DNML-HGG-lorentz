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
import torch.multiprocessing as multi
from functools import partial
from scipy.io import mmread
import os


RESULTS="results"

def calc_metrics_realworld(dataset_name, device_idx, model_n_dim):

    adj_mat = np.load('dataset/' + dataset_name + '/' + dataset_name + '.npy')
    positive_samples = np.load(
        'dataset/' + dataset_name + '/positive_samples.npy')
    negative_samples = np.load(
        'dataset/' + dataset_name + '/negative_samples.npy')
    train_graph = np.load('dataset/' + dataset_name + '/train_graph.npy')
    lik_data = np.load('dataset/' + dataset_name + '/lik_data.npy')

    # print(adj_mat)
    print("n_nodes:", len(adj_mat))
    print("n_edges:", np.sum(adj_mat))
    n_nodes = len(adj_mat)

    params_dataset = {
        'n_nodes': n_nodes,
        'R': np.log(n_nodes) - 0.5,
    }

    # パラメータ
    burn_epochs = 1
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

    basescore_y_and_z, basescore_y_given_z, basescore_z, DNML_codelength, pc_first, pc_second, AIC_naive, BIC_naive, AUC, _, _ = LinkPrediction(
        adj_mat=adj_mat,
        train_graph=train_graph,
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        lik_data=lik_data,
        x_lorentz=None,
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
        calc_groundtruth=False
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

    result["model_n_dims"] = [model_n_dim]
    result["n_nodes"] = [params_dataset["n_nodes"]]
    # result["n_dim"] = params_dataset["n_dim"]
    result["R"] = [params_dataset["R"]]
    # result["sigma"] = params_dataset["sigma"]
    # result["beta"] = params_dataset["beta"]
    result["DNML_codelength"] = DNML_codelength_list
    result["AIC_naive"] = AIC_naive_list
    result["BIC_naive"] = BIC_naive_list
    result["AUC"] = AUC_list
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

    os.makedirs(RESULTS+"/" + dataset_name, exist_ok=True)

    result.to_csv(RESULTS+"/" + dataset_name + "/result_" +
                  str(model_n_dim) + ".csv", index=False)


def data_generation(dataset_name):
    # データセット生成
    edges_ids = np.loadtxt('dataset/' + dataset_name +
                           "/" + dataset_name + ".txt", dtype=int)

    ids_all = set(edges_ids[:, 0]) & set(edges_ids[:, 1])
    n_nodes = len(ids_all)
    adj_mat = np.zeros((n_nodes, n_nodes))
    ids_all = list(ids_all)

    for i in range(len(edges_ids)):
        print(i)
        u = np.where(ids_all == edges_ids[i, 0])[0]
        v = np.where(ids_all == edges_ids[i, 1])[0]
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1

    adj_mat = adj_mat.astype(np.int)
    print("n_nodes:", n_nodes)

    np.save('dataset/' + dataset_name + "/" + dataset_name + ".npy", adj_mat)

    params_dataset = {
        "n_nodes": n_nodes
    }

    positive_samples, negative_samples, train_graph, lik_data = create_test_for_link_prediction(
        adj_mat, params_dataset)

    np.save('dataset/' + dataset_name +
            "/positive_samples.npy", positive_samples)
    np.save('dataset/' + dataset_name +
            "/negative_samples.npy", negative_samples)
    np.save('dataset/' + dataset_name + "/train_graph.npy", train_graph)
    np.save('dataset/' + dataset_name + "/lik_data.npy", lik_data)


if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    import argparse

    parser = argparse.ArgumentParser(description='HGG')
    parser.add_argument('dataset', help='dataset')
    # parser.add_argument('n_dim', help='n_dim')
    parser.add_argument('device', help='device')
    args = parser.parse_args()
    print(args)

    if int(args.dataset) == 0:
        dataset_name = "ca-AstroPh"
    elif int(args.dataset) == 1:
        dataset_name = "ca-HepPh"
    elif int(args.dataset) == 2:
        dataset_name = "ca-GrQc"
    elif int(args.dataset) == 3:
        dataset_name = "ca-CondMat"

    model_n_dims = [2, 4, 8, 16, 32, 64]

    data_generation(dataset_name)

    for model_n_dim in model_n_dims:
        calc_metrics_realworld(dataset_name=dataset_name, device_idx=int(
        args.device), model_n_dim=model_n_dim)
