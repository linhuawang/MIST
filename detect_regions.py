import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import trange
from time import time
from joblib import load
from scipy.stats import ttest_ind
import sys
from Data import Data
import neighbors
from sklearn.metrics import adjusted_rand_score, rand_score

def region_assign(data, epi, min_region=5):
    ccs = neighbors.spatialCCs(data.nodes, data.cormat, epi)
    ccs = [[c.name for c in cc] for cc in ccs]
    
    nb_df = assign_membership(ccs, min_region)
    region_df = nb_df.loc[nb_df['region_size'] > min_region,:]
    
    regions = [cc for cc in ccs if len(cc) == min_region]
    
    for cc1 in ccs:
        if len(cc1) > min_region:
            for cc2 in ccs:
                if len(cc2) > min_region:
                    test_res =  test_identity(cc1, cc2, data.cormat, epi)
                    if test_res:
                        cc1_ind = region_df.loc[cc1, "cluster_ind"].tolist()[0]
                        cc2_ind = region_df.loc[cc2, "cluster_ind"].tolist()[0]
                        region_df.loc[region_df.cluster_ind == cc2_ind, "cluster_ind"] = cc1_ind
                        
    region_df['spot'] = region_df.index.tolist()
    region_grps  = region_df.groupby("cluster_ind")
    
    regions = []
    for _, rg in region_grps:
        regions.append(rg.spot.tolist())
        
    region_df2 = assign_membership(regions, min_region)
    region_df2 = region_df2.loc[region_df.index,:]
    region_df2['region_ind'] = region_df['cluster_ind'].tolist()
    return region_df2, regions

def test_identity(cc1, cc2, cormat, ep, min_region=5):
    #zero out diag of upper traingle of the cormat
    s1 = np.ravel(np.tril(cormat.loc[cc1, cc1], -1))
    #get lower triangle elements of cormat
    s1 = s1[np.where(s1)]
    s2 = np.ravel(np.tril(cormat.loc[cc2, cc2], -1))
    s2 = s2[np.where(s2)]
    s12 = np.ravel(cormat.loc[cc1, cc2])
    
    mean_s1 = np.mean(s1)
    mean_s2 = np.mean(s2)
    mean_s12 = np.mean(s12)
    
    diff1 = mean_s12 - mean_s1
    if len(cc1) > min_region:
        pval1 = ttest_ind(s1, s12)[1]
    else:
        pval1 = 0
        
    diff2 = mean_s12 - mean_s2
    if len(cc2) > min_region:
        pval2 = ttest_ind(s2, s12)[1]
    else:
        pval2 = 0
    mean_pass = (np.absolute(mean_s1 - mean_s12) <= 0.05) and (np.absolute(mean_s2 - mean_s12) <= 0.05)
    p_pass = (pval1 > 0.01) or (pval2 > 0.01)
    max_pass = (np.max(s12) > ep)
    result = (mean_pass or p_pass) and max_pass
    return result

def assign_membership(ccs, min_region=5):
    dfs = []
    k = 0
    for cc in ccs:
        xs = [int(c.split("x")[0]) for c in cc]
        ys = [int(c.split("x")[1]) for c in cc]
    
        df = pd.DataFrame({'x':xs, 'y':ys}, index = cc)
        df['region_size'] = len(cc)
        if len(cc) <  min_region:
            df['cluster_ind'] = -1
        else:
            df['cluster_ind'] = k
            k += 1
        dfs.append(df)
    if len(dfs) > 0:
        dfs = pd.concat(dfs)
    else:
        dfs = pd.DataFrame(columns=['x','y', 'cluster_ind'])
    return dfs

def obj_scores(ccs, cormat, min_region=5, sigma=1, region_min=3):
    #zero out diag of upper traingle of the cormat  
    term1 = 0
    n1 = 0
    for cc1 in ccs:
        if len(cc1) >= min_region: 
            s1 = np.ravel(np.tril(cormat.loc[cc1, cc1], -1))
            #get lower triangle elements of cormat
            s1 = s1[np.where(s1)]
            n1 += len(s1)
            term1 += np.sum(s1)
#             mean_s1 = np.mean(s1)
#             term1 += len(cc1) * mean_s1
    if n1 > 0:
        term1 = term1 / n1
    
    term2 = 0
    n2 = 0
    for cc1 in ccs:
        if len(cc1) >= min_region: 
            for cc2 in ccs:
                if len(cc2) >= min_region and (cc1[0] not in cc2):
                    s12 = np.ravel(cormat.loc[cc1, cc2])
                    n2 += len(s12)
                    term2 += np.sum(s12)

    if n2 > 0:
        term2 = term2 / n2
    
    n_regional = 0
    n_total = cormat.shape[0]
    for cc in ccs:
        if len(cc) >= min_region:
            n_regional += len(cc)
    coverage = n_regional / n_total
    # scores = -1 * (term1 - term2 + sigma * term3) + 1
    #print(term1, term2, term3, scores)

    scores = term2 - term1 - 0.1 * coverage

    if (len(ccs) < region_min) or (coverage < sigma):
        return 1
    return scores

def select_epsilon(data, min_sim=0.4, max_sim=0.91, gap=0.02, 
        min_region=5, sigma=0.5, region_min=3, ep=None):
    st = time()
    
    if ep is None:
        eps = np.arange(min_sim, max_sim, gap)
        scores = []
        for i in trange(len(eps)):
            ep = eps[i]
            _, ccs = region_assign(data, ep,min_region)
            score = obj_scores(ccs, data.cormat, min_region, sigma, region_min)
            scores.append(score)
        
        ind = np.argmin(scores)
        ep, score = eps[ind], scores[ind]
        f, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))
        ax1.plot(eps, scores)
        ax1.vlines(ep, ymin=score, ymax=np.max(scores), color='red', ls='--')
        ax1.set_ylim(score-0.1, np.max(scores) + 0.1)
        region_df, ccs = region_assign(data, ep, min_region)
        end = time()
        print("Epsilon %.3f is selected in %.2f seconds." %(ep, end-st))
        return {'thre_figure': f, 'region_df': region_df, 'threshold': ep}
    else:
        region_df, ccs = region_assign(data, ep, min_region)
        return {'thre_figure': None, 'region_df': region_df, 'threshold': ep}
