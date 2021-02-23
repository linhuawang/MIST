import pandas as pd
import numpy as np
import sys
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, roc_auc_score, recall_score, precision_score
import utils
from time import time
from os.path import exists
from os import mkdir
from spImpute import spImpute
from multiprocessing import Pool
import neighbors
from matplotlib import pyplot as plt

def evaluate_ep(param):
	ho_data, meta_data, ep = param
	imputed = spImpute(ho_data, meta_data, ep)
	return imputed

def epislon_perf(ho_data, ho_mask, meta_data, ori_data, cv_fold, n=1):
	pre_eps = np.arange(0., 1, 0.1)
	eps = []

	cor_mat = utils.spot_PCA_sims(ori_data)
	nodes = neighbors.construct_graph(meta_data)
	cc_lens = []
	for ep in np.flip(pre_eps):
		ccs = neighbors.spatialCCs(nodes, cor_mat, ep)
		if len(ccs) in cc_lens:
			pass
		else:
			cc_lens.append(len(ccs))
			eps.append(ep)

	params = []
	for ep in eps:
		params.append([ho_data, meta_data, ep])
	p = Pool(n)
	imputed = p.map(evaluate_ep, params)
	p.close()
	perf_dfs = []
	for ep, im in zip(eps, imputed):
		perf_df = utils.wholeSlidePerformance(ori_data, ho_mask, ho_data, im, ep)
		perf_dfs.append(perf_df)
	perf_dfs = pd.concat(perf_dfs)
	perf_dfs = perf_dfs.sort_values("PCC")
	perf_dfs["cv_fold"] = cv_fold
	return perf_dfs


def generate_cv_masks(original_data, k=2):
	np.random.seed(2021)
	# make a dumb hold out mask data
	ho_mask = pd.DataFrame(columns = original_data.columns,
								index = original_data.index,
								data = np.zeros(original_data.shape))
	# make templates for masks and ho data for k fold cross validation
	ho_masks = [ho_mask.copy() for i in range(k)]
	ho_dsets = [original_data.copy() for i in range(k)]
	# for each gene, crosss validate
	for gene in original_data.columns.tolist():
		nonzero_spots = original_data.index[original_data[gene] > 0].tolist()
		np.random.shuffle(nonzero_spots)
		nspots = len(nonzero_spots)
		foldLen = int(nspots/k)
		for i in range(k):
			if i != (k-1):
				spot_sampled = nonzero_spots[i*foldLen: (i+1)*foldLen]
			else:
				spot_sampled = nonzero_spots[i*foldLen: ]
			ho_masks[i].loc[spot_sampled, gene] = 1
			ho_dsets[i].loc[spot_sampled, gene] = 0
	return ho_dsets, ho_masks

def select_ep(raw_fn, k=2, n=1):
	start_time = time()
	original_data, meta_data = utils.read_ST_data(raw_fn)
	print(original_data.shape)
	original_data = utils.filterGene_sparsity(original_data,0.8)
	print(original_data.shape)
	original_data =  utils.cpm_norm(original_data, log=False)
	# generate k fold cross validation datasets
	ho_dsets, ho_masks = generate_cv_masks(original_data)
	perf_dfs = []
	for fd in range(2):
		ho_data, ho_mask = ho_dsets[fd], ho_masks[fd]
		perf_df = epislon_perf(ho_data, ho_mask, meta_data, original_data, fd, n)
		perf_dfs.append(perf_df)
	perf_dfs = pd.concat(perf_dfs)
	print(perf_dfs)
	PCC_dfs = perf_dfs.loc[:, ["ModelName", "PCC"]]
	median_pcc_dfs = PCC_dfs.groupby("ModelName").median()
	print(median_pcc_dfs)
	print(median_pcc_dfs.max().index.tolist()[0])
	ep = median_pcc_dfs.max().index.tolist()[0]
	end_time = time()
	print("Epislon %.2f is selected in %.1f seconds." %(ep, end_time - start_time))
	return ep


if __name__ == "__main__":
	raw_fn = sys.argv[1]
	if len(sys.argv) > 2:
		nCore = int(sys.argv[2])
	else:
		nCore = 1
	select_ep(raw_fn, k=2, n=nCore)

