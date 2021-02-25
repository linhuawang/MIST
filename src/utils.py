import pandas as pd
import numpy as np
import os
import sys
from glob import glob
from multiprocessing import Pool
from time import time as TIME
#from logCPM import logCPM
import argparse
from statsmodels.distributions.empirical_distribution import ECDF
import math
from scipy.stats import pearsonr
from scipy.spatial import distance
from matplotlib import pyplot as plt
from scipy.stats import percentileofscore as percentile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def data_norm(data, method='median'):
    assert method in ['median', 'cpm', 'logCPM']
    if method == "median":
        med_libSize =  data.sum(axis=1).median()
        data = data.apply(lambda x: x * med_libSize/ x.sum() , axis=1)
    elif method == "logCPM":
        data = data + 1
        data = data.apply(lambda x: x * (10 ** 6)/ x.sum() , axis=1)
        data = np.log2(data)
    else:
        data = data.apply(lambda x: x * (10 ** 6)/ x.sum() , axis=1)
    return data

# Input: count matrix, rows are samples, columns are genes
def filterGene_std(count_matrix, std_ratio=0.5):
    top_count = int(count_matrix.shape[1] * std_ratio)
    genes = count_matrix.std().sort_values(ascending=False)[:top_count].index
    return count_matrix.loc[:,genes]

def filterGene_sparsity(count_matrix, sparsity_ratio=0.5):
    gene_sparsity = (count_matrix!=0).sum().divide(count_matrix.shape[0])
    genes = gene_sparsity[gene_sparsity >=  sparsity_ratio].index
    return count_matrix.loc[:,genes]

def spot_PCA_sims(count_matrix, method="pearson"):
    np.random.seed(2021)
    assert method in ['pearson', 'kendall', 'spearman']
    count_mat_norm = StandardScaler().fit_transform(count_matrix)
    pca = PCA(n_components=20)
    pca_results = pca.fit_transform(count_mat_norm)
    pca_df = pd.DataFrame(data=pca_results,
                         index=count_matrix.index)
    spot_cors = pd.DataFrame(pca_df.transpose()).corr(method=method)
    return spot_cors

def spot_euc2_aff(slide_meta):
    sample_ids = slide_meta.index.tolist()
    xs = slide_meta.iloc[:,0].to_numpy()
    ys = slide_meta.iloc[:,1].to_numpy()
    nspot = len(sample_ids)
    aff_mat = np.zeros((nspot, nspot))
    for i in range(nspot):
            affs = []
            for j in range(nspot):
                if i == j:
                    sim = 0
                else:
                    dist_square = np.power(xs[i]-xs[j], 2) + np.power(ys[i]-ys[j], 2)
                    sim = 1/ (dist_square + 1)
                affs.append(sim)
            aff_mat[i,:] = affs
    aff_df = pd.DataFrame(data=aff_mat, columns = sample_ids, index= sample_ids)
    return aff_df

def spotPerformance(ori, mask, meta, model_data, model_name):
    spots = ori.index.tolist()
    meta = meta.loc[spots,:]
    rmses = []
    for spot in spots:
        genes = mask.columns[mask.loc[spot,:] == 1].tolist()
        tru = ori.loc[spot, genes].to_numpy()
        imp = model_data.loc[spot, genes].to_numpy()
        rmse = np.sqrt(np.mean(np.square(imp - tru)))
        rmses.append(rmse)
    spot_perf = pd.DataFrame({"x":meta.iloc[:,0],
                                "y":meta.iloc[:,1],
                                "rmse":rmses,
                                "model":model_name})
    return spot_perf

def wholeSlidePerformance(ori, mask, ho, model_data, model_name):
    M = np.ravel(mask)
    inds = np.where(M)
    tru = np.ravel(ori.values)[inds]
    imp = np.ravel(model_data.values)[inds]
    snr = np.log2(np.sum(imp) / np.sum(np.absolute(tru-imp)))
    rmse = np.sqrt(np.mean(np.square(imp - tru)))
    mape = np.mean(np.divide(np.absolute(imp - tru), tru))
    pcc = pearsonr(tru, imp)[0]
    MR1 = float((ho == 0).sum().sum()) / np.prod(ho.shape)
    MR2 = float((model_data == 0).sum().sum()) / np.prod(model_data.shape)
    perf_df = pd.DataFrame(data=[[rmse, mape, snr, pcc, model_name, MR1, MR2, MR1-MR2]],
                          columns= ['RMSE', 'MAPE', 'SNR', 'PCC', 'ModelName', 'hoMR', 'impMR', 'redMR'])
    return perf_df


def spot_density(count_matrix):
    densities = np.array([])
    for spot in count_matrix.index:
        zeroNum = (count_matrix.loc[spot,:] == 0).sum()
        density = 1 - float(zeroNum) / count_matrix.shape[1]
        densities = np.append(densities, density)
    return densities

def gene_density(count_matrix):
    densities = np.array([])
    for gene in count_matrix.columns:
        zeroNum = (count_matrix.loc[:, gene] == 0).sum()
        density = 1 - float(zeroNum) / count_matrix.shape[0]
        densities = np.append(densities, density)
    sparsities = 1 - densities
    return densities, sparsities


def count_density(count_matrix):
    num_zeros = (count_matrix != 0).sum().sum()
    return float(num_zeros) / (count_matrix.shape[0] * count_matrix.shape[1])

def spot_read_counts(count_matrix):
    read_counts = np.array([])
    for spot in count_matrix.index:
        read_counts = np.append(read_counts, count_matrix.loc[spot,:].sum())
    return read_counts

def gene_read_counts(count_matrix):
    read_counts = np.array([])
    for gene in count_matrix.columns:
        read_counts = np.append(read_counts, count_matrix.loc[:, gene].sum())
    return read_counts

def read_ST_data(count_fn, sep=","):
    data = pd.read_csv(count_fn, sep=",", index_col=0)
    data = data.fillna(0)
    meta = pd.DataFrame({"coordX": [int(i.split("x")[0]) for i in data.index.tolist()],
                         "coordY": [int(i.split("x")[1]) for i in data.index.tolist()]},
                         index = data.index)
    return data, meta

if __name__ == "__main__":
    count_matrix = pd.read_csv("/Users/linhuaw/Documents/STICK/results/mouse_wt/logCPM.csv", index_col=0)
    #print(count_matrix.shape)
    #count_matrix = filterGene_sparsity(count_matrix, 0.4)
    #print(count_matrix.shape)
    cors = spot_PCA_sims(count_matrix)
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.heatmap(data=cors, vmin=0, vmax=1, cmap="Blues")
    plt.show()
    #plt.close()
    cors2 = spot_exp_sims(count_matrix)
    sns.heatmap(data=cors2, vmin=0, vmax=1, cmap="Blues")
    plt.show()
