import scanpy as sc 
import pandas as pd
import numpy as np
from sklearn import metrics
import torch
import random

positive_coexpression_pairs = [
    ['HUMAN_CD3E', 'HUMAN_CD4'],
    ['HUMAN_CD3D', 'HUMAN_CD8A'],
    ['HUMAN_PTPRC', 'HUMAN_CD68'],
    ['HUMAN_CD19', 'HUMAN_MS4A1']
]

negative_coexpression_pairs = [
    ['HUMAN_CD3D', 'HUMAN_CD68'],
    ['HUMAN_CD68', 'HUMAN_MS4A1']
]

# This function loads CITE-seq dataset.
def load_data():

    # Load data
    adata = sc.read_h5ad("anndata_citseq_rnaseq.h5ad")
    surface = sc.read_h5ad("anndata_citseq_surface_ab.h5ad")
    df_clusters = surface.obs[["seurat_clusters"]]
    df_clusters.index = df_clusters.index.set_names('cell_id')
    surface_clusters = list(df_clusters.seurat_clusters[adata.obs_names])
    adata.obs['target'] = surface_clusters

    # Filter genes
    rs = adata.X.sum(axis=0)
    adata = adata[:, rs > 100.]

    # Subsample
    seed=10
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(a=seed)

    sc.pp.subsample(adata, n_obs=1000)
    print(f"max in subsampled adata: {adata.X.max()}")
    print(f"shape of adata: {adata.shape}")

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    Y = adata.X.copy()

    # Compute PCA
    sc.tl.pca(adata, svd_solver='arpack')
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
    return df[[p1,p2]].corr()[p1][p2]

# This function computes the clustering for a given proportion of 
# HVGs in a scRNA-seq dataset.
def get_clustering(adata, proportion, n_neighbors=10, n_pcs=40, resolution=0.8):
    # Compute number of genes to retain
    n_top_genes = int(np.round(adata.n_vars * proportion))

    # Find desired number of HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    hvgs = adata.var.highly_variable.tolist()
    print(f"Computed {sum(hvgs)} HVGs.")
    # Subset adata to HVGs
    adata = adata[:,adata.var.highly_variable]

    # Compute PCA and neighbour graph
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # Cluster
    sc.tl.leiden(adata, resolution=resolution)
    return adata, hvgs

def cluster_test(adata, hvgs, n_neighbors=10, n_pcs=40, resolution=0.8):
    print(f"Got passed in {sum(hvgs)} HVGs")
    print(f"Test dataset is {adata.shape}")
    # Subset adata to HVGs
    adata = adata[:,hvgs]

    # Compute PCA and neighbour graph
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # Cluster
    sc.tl.leiden(adata, resolution=resolution)
    return adata

# This function computes the clustering for a given propotion 
# of HVGs retained in scRNA-seq data and computes objective values for it 
# (supervised objectives are # w.r.t. labels from surface protein expression 
# clustering).
def true_f(proportions, adata, Y):
    results = {}
    from collections import defaultdict
    results_list = defaultdict(list)

    ari = [] 
    nmi = []
    sil = []
    cal = []
    db = []
    
    for prop in proportions:        
        results[prop] = {}
        clustered, hvgs = get_clustering(adata.copy(), prop)

        # Supervised metrics
        ari.append(metrics.adjusted_rand_score(adata.obs.target.astype(str), clustered.obs.leiden))
        nmi.append(metrics.normalized_mutual_info_score(adata.obs.target.astype(str), clustered.obs.leiden))

        # Unsupervised metrics
        sil.append(metrics.silhouette_score(adata.obsm['X_pca'], clustered.obs.leiden).astype(np.float32))
        cal.append(metrics.calinski_harabasz_score(adata.obsm['X_pca'], clustered.obs.leiden).astype(np.float32))
        db.append(-metrics.davies_bouldin_score(adata.obsm['X_pca'], clustered.obs.leiden).astype(np.float32))
        
        # Coexpression metrics
        unique_clusters = np.unique(clustered.obs.leiden)
        cluster_means = np.concatenate([Y[clustered.obs.leiden == cl,:].mean(0).reshape(1,-1) for cl in unique_clusters], axis=0)
 
        for pair in positive_coexpression_pairs:
            pair_str = pair[0] + "_" + pair[1] + "_+"
            results[prop][pair_str] = get_coexpression(cluster_means, pair[0], pair[1], adata.var_names).astype(np.float32)

        for pair in negative_coexpression_pairs:
            pair_str = pair[0] + "_" + pair[1] + "_-"
            results[prop][pair_str] = -get_coexpression(cluster_means, pair[0], pair[1], adata.var_names).astype(np.float32)
            
        for k,v in results[prop].items():
            results_list[k].append(v)
    
    pos_pairs = []
    for pair in positive_coexpression_pairs:
        pair_str = pair[0] + "_" + pair[1] + "_+"
        obj = np.array(results_list[pair_str])
        pos_pairs.append(torch.reshape(torch.tensor([obj]), (obj.shape[0], 1))) 

    neg_pairs = []
    for pair in negative_coexpression_pairs:
        pair_str = pair[0] + "_" + pair[1] + "_-"
        obj = np.array(results_list[pair_str])
        neg_pairs.append(torch.reshape(torch.tensor([obj]), (obj.shape[0], 1))) 

    ari = np.array(ari)
    nmi = np.array(nmi)
    sil = np.array(sil)
    cal = np.array(cal)
    db = np.array(db)

    ari = torch.reshape(torch.tensor([ari]), (ari.shape[0], 1))
    nmi = torch.reshape(torch.tensor([nmi]), (nmi.shape[0], 1))
    sil = torch.reshape(torch.tensor([sil]), (sil.shape[0], 1))
    cal = torch.reshape(torch.tensor([cal]), (cal.shape[0], 1))
    db = torch.reshape(torch.tensor([db]), (db.shape[0], 1))
        
    y = torch.cat([sil, cal, db] + pos_pairs + neg_pairs, axis=1)
    return y, ari, nmi, hvgs

def probe_test(proportions, adata, Y, hvgs):
    results = {}
    from collections import defaultdict
    results_list = defaultdict(list)

    ari = [] 
    nmi = []
    
    for prop in proportions:        
        clustered = cluster_test(adata.copy(), hvgs)

        # Supervised metrics
        ari.append(metrics.adjusted_rand_score(adata.obs.target.astype(str), clustered.obs.leiden))
        nmi.append(metrics.normalized_mutual_info_score(adata.obs.target.astype(str), clustered.obs.leiden))
        print(f"ARI on test dataset {adata.shape} is {ari}")

    ari = np.array(ari)
    nmi = np.array(nmi)

    ari = torch.reshape(torch.tensor([ari]), (ari.shape[0], 1))
    nmi = torch.reshape(torch.tensor([nmi]), (nmi.shape[0], 1))
        
    return ari, nmi

def get_labels():
    labels = []
    labels.append('Sil')
    labels.append('Cal')
    labels.append('Db')
    for pair in positive_coexpression_pairs:
        labels.append(pair[0] + "_" + pair[1] + "_+")

    for pair in negative_coexpression_pairs:
        labels.append(pair[0] + "_" + pair[1] + "_-")
    return labels
