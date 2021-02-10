import pandas as pd
import numpy as np
import os
import sys
from glob import glob
from multiprocessing import Pool
from time import time
import argparse
from statsmodels.distributions.empirical_distribution import ECDF
import math
from scipy.stats import pearsonr
from scipy.spatial import distance
from progressbar import *
import sys
from neighbors import *
from shutil import rmtree
from scipy.sparse import csgraph
import cvxpy as cp
from tqdm import tqdm, trange
import utils

def spImpute(data, meta, epislon=0.6, n=1): # multiprocessing not implemented yet
	#start_time = time()
	cor_mat = spot_PCA_sims(data)
	nodes = construct_graph(meta)
	ccs = spatialCCs(nodes, cor_mat, epislon)
	f = plot_ccs(ccs, meta, "epsilon = %.2f" %epislon)
	spots = data.index.tolist()
	known_idx = np.where(data.values)
	imputed = data.copy()

	for i1 in range(len(ccs)):
		cc = ccs[i1]
		cc_spots = [c.name for c in cc]
		other_spots = [s for s in spots if s not in cc_spots]
		m = len(cc_spots)
		s = int(len(other_spots) / 10)
		values = np.zeros(shape=(m, data.shape[1], 10))
		for i2 in range(10): 
			np.random.seed(i2)
			sampled_spot = list(np.random.choice(other_spots, s, replace=False))
			ri_spots = cc_spots + sampled_spot
			ri_impute = rankMinImpute(data.loc[ri_spots,:])
			values[:,:,i2] = ri_impute.values[:m, :]
			print(i1, i2)
		imputed.loc[cc_spots,:] = np.mean(values, axis=2)

	assert np.all(data.values[known_idx] > 0)
	imputed.values[known_idx] = data.values[known_idx]

	#end_time = time()
	#print("Imptuation finished in %.2f seconds." %(end_time - start_time))
	return imputed, f

def spImpute_with_corMat(data, meta, cor_mat, epislon=0.6, n=1): # multiprocessing not implemented yet
	#start_time = time()
	nodes = construct_graph(meta)
	ccs = spatialCCs(nodes, cor_mat, epislon)
	spots = data.index.tolist()
	known_idx = np.where(data.values)
	imputed = data.copy()

	for i1 in range(len(ccs)):
		cc = ccs[i1]
		cc_spots = [c.name for c in cc]
		other_spots = [s for s in spots if s not in cc_spots]
		m = len(cc_spots)
		s = int(len(other_spots) / 10)
		values = np.zeros(shape=(m, data.shape[1], 10))
		for i2 in range(10): 
			np.random.seed(i2)
			sampled_spot = list(np.random.choice(other_spots, s, replace=False))
			ri_spots = cc_spots + sampled_spot
			ri_impute = rankMinImpute(data.loc[ri_spots,:])
			values[:,:,i2] = ri_impute.values[:m, :]
			print(i1, i2)
		imputed.loc[cc_spots,:] = np.mean(values, axis=2)

	assert np.all(data.values[known_idx] > 0)
	imputed.values[known_idx] = data.values[known_idx]
	return imputed


def rankMinImpute(data):
	good_genes = data.columns[((data > 3).sum() > 2)].tolist()
	bad_genes = data.columns[((data > 3).sum() <= 2)].tolist()
	all_genes = data.columns.tolist()
	bad_data = data.loc[:, bad_genes]
	good_data = data.loc[:, good_genes]

	t1 = time()
	D = np.ravel(good_data.values) # flattened count matrix
	idx = np.where(D) # nonzero indices of D
	y = D[idx] # nonzero observed values 
	n = np.prod(good_data.shape)
	err= 1E-12
	x_initial = np.zeros(np.prod(good_data.shape))
	tol= 1E-4
	decfac = 0.7
	alpha = 1.1
	x = x_initial
	lamIni = decfac * np.max(np.absolute(y))
	lam = lamIni

	f1 = np.linalg.norm(y - x[idx], 2) + np.linalg.norm(x, 1) * lam

	while lam > lamIni * tol:
		for i in range(20):
			f0 = f1
			z = np.zeros(n) 
			z[idx] = y - x[idx]
			b = x + (1/alpha) * z
			B = np.reshape(b, good_data.shape)
			U, S, V = np.linalg.svd(B,full_matrices=False)
			s = softThreshold(S, lam/(2*alpha))
			X = U @ np.diag(s) @ V
			X[X<0] = 0
			x = X.ravel()
			f1 = np.linalg.norm(y - x[idx], 2) + np.linalg.norm(x, 1) * lam
			e1 = np.linalg.norm(f1-f0)/np.linalg.norm(f1+f0)			
			if e1 < tol:
				break
		e2 = np.linalg.norm(y-x[idx])
		if e2 < err:
			break
		lam = decfac*lam
		#print("lambda: %.2f/%.2f, error: %.4f/%.4f" %(lam,lamIni * tol, e2, err))
	imputed = pd.DataFrame(data=X, index=bad_data.index, columns=good_data.columns)
	imputed = pd.concat([bad_data, imputed], axis=1)
	assert not imputed.isna().any().any()
	imputed = imputed.loc[data.index, all_genes]
	t2 = time()
	#print("Rank minmization imputation for %d spots finished in %.1f seconds." %(data.shape[0],t2-t1))
	return imputed

def softThreshold(s, l):
	return np.multiply(np.sign(s), np.absolute(s - l))

def run_impute(param):
	ho_data, meta_data, ep = param
	imputed = spImpute_with_corMat(ho_data, meta_data, cor_mat, ep)
	return imputed

def epislon_perf(ho_data, ho_mask, meta_data, ori_data, cor_mat, cv_fold, n=1):
	pre_eps = np.arange(0.2, 1.0, 0.1)
	eps = []

	nodes = construct_graph(meta_data)
	cc_lens = []
	for ep in np.flip(pre_eps):
		ccs = spatialCCs(nodes, cor_mat, ep)
		if len(ccs) in cc_lens:
			pass
		else:
			cc_lens.append(len(ccs))
			eps.append(ep)

	params = []
	for ep in eps:
		params.append([ho_data, meta_data, cor_mat, ep])

	p = Pool(n)
	imputed = p.map(run_impute, params)
	p.close()
	perf_dfs = []
	ori_data = np.log2(ori_data + 1)
	ho_data = np.log2(ho_data + 1)
	for ep, im in zip(eps, imputed):
		im = np.log2(im + 1)
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

def select_ep(original_data, meta_data, k=2, n=1):
	start_time = time()
	original_data =  utils.cpm_norm(original_data, log=False)
	cor_mat = spot_PCA_sims(original_data)
	print(original_data.shape)
	training_data = utils.filterGene_sparsity(original_data,0.8)
	print(training_data.shape)
	
	# generate k fold cross validation datasets
	ho_dsets, ho_masks = generate_cv_masks(training_data)
	perf_dfs = []
	for fd in range(2):
		ho_data, ho_mask = ho_dsets[fd], ho_masks[fd]
		perf_df = epislon_perf(ho_data, ho_mask, meta_data, training_data, cor_mat, fd, n)
		perf_dfs.append(perf_df)
	perf_dfs = pd.concat(perf_dfs)
	print(perf_dfs)
	PCC_dfs = perf_dfs.loc[:, ["ModelName", "PCC"]]
	median_pcc_dfs = PCC_dfs.groupby("ModelName").median()
	median_pcc_dfs['epislon'] = median_pcc_dfs.index.to_numpy()
	ep = median_pcc_dfs.loc[median_pcc_dfs.PCC == median_pcc_dfs.PCC.max(),"epislon"].tolist()[0]
	print(ep)
	end_time = time()
	print("Epislon %.2f is selected in %.1f seconds." %(ep, end_time - start_time))
	return ep

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Algorithm variables.')
	parser.add_argument('-f', '--countpath', type=str,
                    help='path to the input count matrix,\
                    1st row is the gene name and 1st column is spot ID')
	parser.add_argument('-e', '--epislon', type=float, default=0.6,
                    help='threshold to filter low confident edges')
	parser.add_argument('-o', '--out_fn', type=str, default='none',
                    help='File path to save the imputed values.')
	parser.add_argument('-m', '--member_fn', type=str, default='none',
                    help='File path to save the other files.')
	parser.add_argument('-s', '--select', type=int, default=1,
					help = 'whether infer epislon using holdout test or not.')
	parser.add_argument('-n', '--ncore', type=int, default=1,
					help = 'number of processors')

	args = parser.parse_args()
	count_fn = args.countpath
	epi = args.epislon
	out_fn = args.out_fn
	ncore = args.ncore
	select = args.select
	count_matrix, meta_data = read_ST_data(count_fn)
	count_matrix = count_matrix.fillna(0)
	if select == 1: # takes hours even with multiprocessing
		ep = select_ep(count_matrix, meta_data, k=2, n=ncore)
	else:
		ep = epi
	count_matrix = utils.cpm_norm(count_matrix)
	imputed, figure = spImpute(count_matrix, meta_data, ep, ncore)

	# member_fn = args.member_fn
	# imputed = CCRM(count_matrix, meta_data, epi)
	if out_fn != "none":
		imputed.to_csv(out_fn)
		fig_out = out_fn.split(".csv")[0] + ".png"
		figure.savefig(fig_out)
	# # if member_fn != "none":
	# # 	memberships.to_csv(member_fn)

