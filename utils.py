#!/usr/bin/env python
"""Provides helper functions for processing and examining ST data, 
generating hold-out data and evaluating performance.
"""
import pandas as pd
import numpy as np
import os
import sys
from glob import glob
from multiprocessing import Pool
from time import time as TIME
import argparse
from statsmodels.distributions.empirical_distribution import ECDF
import math
from scipy.stats import pearsonr
from scipy.spatial import distance
from matplotlib import pyplot as plt
from scipy.stats import percentileofscore as percentile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import csr_matrix
from statsmodels.stats.weightstats import DescrStatsW 

__author__ = "Linhua Wang"
__license__ = "GNU General Public License v3.0"
__maintainer__ = "Linhua Wang"
__email__ = "linhuaw@bcm.edu"

def evalSlide(ori, mask, ho, model_data, model_name):
    """Method to evaluate the whole-slide-level imputation results
    
    Parameters:
    ----------
    ori: the original observed gene expression data frame
    mask: holdout mapping data frame with the same index and columns as ori.
            value 1 indicates holdout and 0 otherwise.
    ho: the gene expression data frame after removing holdout values
    model_data: the imputed gene expression value
    model_name: the name of the imputation algorithm

    Return:
    ------
    Performance data frame.
    """
    M = np.ravel(mask)
    inds = np.where(M) ## Get the indices of the holdout values in the flattened matrix
    tru = np.ravel(ori.values)[inds] ## Get the ground truth of holdout values
    imp = np.ravel(model_data.values)[inds] ## Get imputed values for holdout part
    snr = np.log2(np.sum(imp) / np.sum(np.absolute(tru-imp))) 
    rmse = np.sqrt(np.mean(np.square(imp - tru))) ## Calculate RMSE values
    mape = np.mean(np.divide(np.absolute(imp - tru), tru)) ## Calculate MAPE values
    pcc = pearsonr(tru, imp)[0] ## Calculate PCC value
    MR1 = float((ho == 0).sum().sum()) / np.prod(ho.shape) ## Calculate missing rates
    MR2 = float((model_data == 0).sum().sum()) / np.prod(model_data.shape) ## Calculate missing rates
    ## Integrate results and generate output data frame
    perf_df = pd.DataFrame(data=[[rmse, mape, snr, pcc, model_name, MR1, MR2, MR1-MR2]],
             columns= ['RMSE', 'MAPE', 'SNR', 'PCC', 'ModelName', 'hoMR', 'impMR', 'redMR'])
    return perf_df

def data_norm(data, method='cpm'):
    """Method to normalize raw gene counts by library size
    
    Parameters:
    ----------
    data: raw count data frame
    method: normalization approach, has to be one of ['none','median', 'cpm', 'logCPM', 'logMed']

    Return:
    data: data frame, normalized gene expression
    libSize: data frame, library size for each spot
    """
    assert method in ['none','median', 'cpm', 'logCPM', 'logMed']
    if method == "none":
        return data
    ## calculate library size for each spot
    libsize = pd.DataFrame({"depth":data.sum(axis=1).to_numpy()},
        index=data.index)
    ## calculate median library size
    med_libSize =  libsize.depth.median()
    ## Normalize data based on defined approach
    if method == "median":
        data = data.apply(lambda x: x * med_libSize/ x.sum() , axis=1)
    elif method == "logMed":
        data = np.log2(1 + data.apply(lambda x: x * med_libSize/ x.sum() , axis=1))
    elif method == "logCPM":
        data = data.apply(lambda x: x * (10 ** 6)/ x.sum() , axis=1)
        data = np.log2(data+1)
    else:
        data = data.apply(lambda x: x * (10 ** 6)/ x.sum() , axis=1).astype(int)
    return data, libsize

def data_denorm(data, libsize, method, libfile=None):
    """Method to de-normalize normalized gene expresion to get raw gene counts.
    
    Parameters:
    ----------
    data: data frame, normalized gene expression
    libsize: data frame, spot library size. libfile has to be provided if libsize is not provided.
    method: normalization approach, has to be one of ['none','median', 'cpm', 'logCPM', 'logMed']
    libfile: path to the library size file. libfile has to be provided if libsize is not provided.

    Return:
    ------
    count: data frame, raw gene expression count for each spot.
    """
    assert method in ['none','median', 'cpm', 'logCPM', 'logMed']

    if libfile != None:
        libsize = pd.read_csv(libfile, index_col=0)

    libsize = libsize.loc[data.index,:] # n*1
    count = data.copy() # n*p

    if method in ['logCPM', 'logMed']:
        count = (np.power(2, count) - 1).astype(int)

    if method in ["median", "logMed"]:
        med_libSize = libsize.depth.median()
        count = (count / med_libSize)
    else:
        count = (count / (10 ** 6))
    count = count.mul(libsize.depth.to_numpy(), axis=0).astype(int)
    return count

def filterGene_std(count_matrix, std_ratio=0.5):
    """Methods to filter genes with high variance (standard deviation)
    
    Parameters:
    ----------
    count_matrix: data frame, normalized gene expression
    std_ratio: top ratio to be kept

    Return:
    ------
    gene expression with high-varaince genes.
    """
    top_count = int(count_matrix.shape[1] * std_ratio)
    genes = count_matrix.std().sort_values(ascending=False)[:top_count].index
    return count_matrix.loc[:,genes]

def filterGene_sparsity(count_matrix, sparsity_ratio=0.5):
    """Methods to filter genes with high coverage (low sparsity)
    
    Parameters:
    ----------
    count_matrix: data frame, normalized gene expression
    sparsity_ratio: sparsity threshold to be kept

    Return:
    ------
    gene expression with high-coverage genes.
    """
    gene_sparsity = (count_matrix!=0).sum().divide(count_matrix.shape[0])
    genes = gene_sparsity[gene_sparsity >=  sparsity_ratio].index
    return count_matrix.loc[:,genes]

def weighted_PCA_sims(count_matrix, n_pcs):
    pca = PCA(n_components=n_pcs)
    pca_res = pca.fit_transform(count_matrix)
    ratio = np.sqrt(pca.explained_variance_ratio_)
    weights = ratio/np.sum(ratio)
    d = DescrStatsW(pca_res.transpose(), weights=weights)
    weighted_corrs = d.corrcoef
    return weighted_corrs, pca_res

def spot_PCA_sims(count_matrix, methods=["spearman"], n_pcs=10):
    """Methods to calculate spot-spot similarity at reduced dimenstion
    Parameters:
    ----------
    count_matrix: data frame, normalized gene expression
    method: correlation measurement method, one of ['pearson', 'kendall', 'spearman']

    Return:
    ------
    spot-spot similarity matrix
    """
    np.random.seed(2022)
    ## Z-score normalize gene expression
    count_mat_norm = StandardScaler(with_mean=False).fit_transform(csr_matrix(count_matrix.values))
    ## Reduce the dimension
    print(f"Extracting top {n_pcs} PCs ...")
    #pca = PCA(n_components=n_pcs)
    #pca_results = pca.fit_transform(count_mat_norm)
    tsvd = TruncatedSVD(n_components=n_pcs, random_state=2022)
    pca_results = tsvd.fit_transform(count_mat_norm)
    pca_df = pd.DataFrame(data=pca_results,
                         index=count_matrix.index)
    print(f"PCA done. Now getting similarities using {methods}.")
    ## Calculate Pearson Correlation Coefficient as the weights for spot-spot edges
    # edited: use average of three metrics
    spot_cors = np.mean([pd.DataFrame(pca_df.transpose()).corr(method=method) for method in methods], axis=0)
    spot_cors = pd.DataFrame(data=spot_cors,
                             index=count_matrix.index,
                             columns=count_matrix.index)
    print("Spot similarity matrix extracted.")
    return spot_cors

def spot_euc2_aff(slide_meta):
    """Methods to calculate spot-spot similarity using eucledian distance
    between the coordinates

    Parameters:
    ----------
    slide_meta: data frame, coordinates of spots

    Return:
    ------
    spot-spot affinity matrix
    """
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
                    # calculate euclidean distance
                    dist_square = np.power(xs[i]-xs[j], 2) + np.power(ys[i]-ys[j], 2) 
                    # convert distance to affinity
                    sim = 1/ (dist_square + 1)
                affs.append(sim)
            aff_mat[i,:] = affs
    aff_df = pd.DataFrame(data=aff_mat, columns = sample_ids, index= sample_ids)
    return aff_df

def spot_density(count_matrix):
    """Methods to calculate spot non-zero proportions
    
    Parameters:
    ----------
    count_matrix: data frame, gene expression

    Return:
    ------
    spot non-zero proportions
    """
    densities = np.array([])
    for spot in count_matrix.index:
        zeroNum = (count_matrix.loc[spot,:] == 0).sum()
        density = 1 - float(zeroNum) / count_matrix.shape[1]
        densities = np.append(densities, density)
    return densities

def gene_density(count_matrix):
    """Methods to calculate gene non-zero proportions
    
    Parameters:
    ----------
    count_matrix: data frame, gene expression

    Return:
    ------
    gene non-zero proportions
    """
    densities = np.array([])
    for gene in count_matrix.columns:
        zeroNum = (count_matrix.loc[:, gene] == 0).sum()
        density = 1 - float(zeroNum) / count_matrix.shape[0]
        densities = np.append(densities, density)
    sparsities = 1 - densities
    return densities, sparsities


def count_density(count_matrix):
    """Methods to calculate whole-matrix non-zero proportions
    
    Parameters:
    ----------
    count_matrix: data frame, gene expression

    Return:
    ------
    non-zero proportions in the entire gene expression matrix
    """
    num_zeros = (count_matrix != 0).sum().sum()
    return float(num_zeros) / (count_matrix.shape[0] * count_matrix.shape[1])

def spot_read_counts(count_matrix):
    """Method to calculate the read counts for each spot
    
    Parameters:
    ----------
    count_matrix: data frame, raw count gene expression

    Return:
    ------
    read_counts: data frame, spot read counts
    """
    read_counts = np.array([])
    for spot in count_matrix.index:
        read_counts = np.append(read_counts, count_matrix.loc[spot,:].sum())
    return read_counts

def gene_read_counts(count_matrix):
    """Method to calculate the read counts for each gene
    
    Parameters:
    ----------
    count_matrix: data frame, raw count gene expression

    Return:
    ------
    read_counts: data frame, gene read counts
    """
    read_counts = np.array([])
    for gene in count_matrix.columns:
        read_counts = np.append(read_counts, count_matrix.loc[:, gene].sum())
    return read_counts

def read_ST_data(count_fn, sep=","):
    """Method to read ST data
    
    Parameters:
    ----------
    count_fn: str, path to the gene expression data.
    sep: str, delimiter of the file format.

    Return:
    ------
    data: data frame, gene expression
    meta: data frame, coordinates of each spot
    """
    data = pd.read_csv(count_fn, sep=",", index_col=0)
    data = data.fillna(0)
    meta = pd.DataFrame({"coordX": [int(i.split("x")[0]) for i in data.index.tolist()],
                         "coordY": [int(i.split("x")[1]) for i in data.index.tolist()]},
                         index = data.index)
    return data, meta

def generate_cv_masks(original_data, genes, k=5):
    """Method to generate k fold cross-validation data
    
    Parameters:
    ----------
    original_data: data frame, gene expression
    genes: list of genes to perform holdout on
    k: number of CV fold

    Return:
    ------
    ho_dsets: list of data frame, each one is a hold-out data
    ho_masks: list of data frame, each one is the hold-out mapping indicator,
            value 1 means holdout, 0 otherwise
    """

    np.random.seed(2021)
    # make a dumb hold out mask data
    ho_mask = pd.DataFrame(columns = original_data.columns,
                                index = original_data.index,
                                data = np.zeros(original_data.shape))

    # make templates for masks and ho data for k fold cross validation
    ho_masks = [ho_mask.copy() for i in range(k)]
    ho_dsets = [original_data.copy() for i in range(k)]

    for gene in genes:
        # extract candidate non-zero spots of every gene for holdout
        nonzero_spots = original_data.index[original_data[gene] > 0].tolist()
        np.random.shuffle(nonzero_spots) # random shuffle the order
        nspots = len(nonzero_spots)
        foldLen = int(nspots/k) # number of values to be holdout
        # Iteratively holdout for K folds
        for i in range(k):
            if i != (k-1):
                spot_sampled = nonzero_spots[i*foldLen: (i+1)*foldLen]
            else:
                spot_sampled = nonzero_spots[i*foldLen: ]
            # Preparing the mapping mask
            ho_masks[i].loc[spot_sampled, gene] = 1
            ho_dsets[i].loc[spot_sampled, gene] = 0
    return ho_dsets, ho_masks



