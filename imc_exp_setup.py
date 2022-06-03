from scanpy import read_h5ad
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import torch
import random

positive_coexpression_pairs = [
    ['ECadherin', 'panCytokeratin'],
    ['CD45', 'CD20'],
    ['CD45', 'CD68'],
    ['CD45', 'CD3'],
    ['Vimentin', 'Fibronectin'],
    ['CD19','CD20'],
    ['panCytokeratin', 'CK5'],

# This function loads jackson-imc dataset.
def load_data():
    # Load data
    adata = read_h5ad("basel_zuri.h5ad")
    adata = adata[adata.obs.index.str.contains("Basel"),:]

    # Subsample
    seed=10
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(a=seed)

    to_sample = np.random.choice(adata.shape[0], size=5000, replace=False)
    adata = adata[to_sample,:]
    print(f"max in subsampled adata: {adata.X.max()}")
    print(f"shape of adata: {adata.shape}")
    # Load clusters
    clusters = pd.read_csv("PG_final_k20.csv").set_index("id")
    gt_clusters = list(clusters.PhenoGraphBasel[adata.obs_names])
    adata.obs['target'] = gt_clusters
    adata = adata[:,~adata.var_names.str.contains("Rutheni|80ArArArAr80Di|Hg202|I127|In115|208PbPb208Di|Xe126|Xe131|Pb20|Xe13")]
    Y = adata.X.copy()
    Y = np.arcsinh(Y)
    return adata, Y

def split_data(adata, Y):
    # Split subsampled data to train and test
    train = np.random.choice(adata.shape[0], size=int(adata.shape[0]*0.7), replace=False)
    adata_train = adata[train,:]
    Y_train = Y[train,:]

    test = set(range(adata.shape[0])).difference(set(train))
    adata_test = adata[list(test),:]
    Y_test = Y[list(test),:]    
    return adata_train, Y_train, adata_test, Y_test

# This function computes correlation between cluster
# mean expression.
def get_coexpression(cluster_means, p1, p2, var_names):
    df = pd.DataFrame(cluster_means, columns=var_names)
    return df.corr()[p1][p2]

# This function arcisnh-normalised the data with
# the given cofactor and clusters it with k-means.
def norm_adata(adata, cofactor=5., n_clusters=10):
    # Normalise data with cofactor
    adata.X = np.arcsinh(adata.X / cofactor)
    km = KMeans(n_clusters, random_state=int(np.random.choice(1000, 1)))
    # Cluster data
    km.fit(adata.X)
    adata.obs['leiden'] = km.labels_
    return adata

# This function computes the clustering resulting from a given cofactor
# and computes objective values for it (supervised objectives are 
# w.r.t. true labels).
def true_f(cofactors, adata, Y):
    results = {} 

    from collections import defaultdict
    results_list = defaultdict(list)
    ari = [] 
    nmi = []
    
    for cofactor in cofactors:        
        results[cofactor] = {}
        adata_norm = norm_adata(adata.copy(), cofactor)

        unique_clusters = np.unique(adata_norm.obs.leiden)
        cluster_means = np.concatenate([Y[adata_norm.obs.leiden == cl,:].mean(0).reshape(1,-1) for cl in unique_clusters], axis=0)
 
        for pair in positive_coexpression_pairs:
            pair_str = pair[0] + "_" + pair[1] + "_+"
            results[cofactor][pair_str] = get_coexpression(cluster_means, pair[0], pair[1], adata_norm.var_names).astype(np.float32)

        ari.append(metrics.adjusted_rand_score(adata_norm.obs.target.astype(str), adata_norm.obs.leiden))
        nmi.append(metrics.normalized_mutual_info_score(adata_norm.obs.target.astype(str), adata_norm.obs.leiden))

        for k,v in results[cofactor].items():
            results_list[k].append(v)
    
    pos_pairs = []
    for pair in positive_coexpression_pairs:
        pair_str = pair[0] + "_" + pair[1] + "_+"
        obj = np.array(results_list[pair_str])
        pos_pairs.append(torch.reshape(torch.tensor([obj]), (obj.shape[0], 1))) 

    ari = np.array(ari)
    nmi = np.array(nmi)
    ari = torch.reshape(torch.tensor([ari]), (ari.shape[0], 1))
    nmi = torch.reshape(torch.tensor([nmi]), (nmi.shape[0], 1))
        
    y = torch.cat(pos_pairs, axis=1)
    
    return y, ari, nmi

def get_labels():
    labels = []
    for pair in positive_coexpression_pairs:
        labels.append(pair[0] + "_" + pair[1] + "_+")

    return labels
