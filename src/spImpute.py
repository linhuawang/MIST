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
import sys
from neighbors import *
from shutil import rmtree
from scipy.sparse import csgraph
from tqdm import tqdm, trange
import utils
import Data

def spImpute(data_obj, nExperts=10): # multiprocessing not implemented yet
	start_time = time()
	data = data_obj.count
	meta = data_obj.meta
	epsilon = data_obj.epsilon
	radius = data_obj.radius
	nodes = data_obj.nodes
	merge = data_obj.merge
	cor_mat = data_obj.cormat
	#nodes = construct_graph(meta, radius)
	ccs = spatialCCs(nodes, cor_mat, epsilon, merge=0)

	imputed_whole = rankMinImpute(data)
	t1 = time()
	print("Base line imputation done in %.1f seconds ..." %(t1  - start_time))
	member_df, f = plot_ccs(ccs, meta, "epsilon = %.2f" %epsilon)

	if len(ccs) == 1:
		return imputed_whole, member_df, f
		
	spots = data.index.tolist()
	known_idx = np.where(data.values)
	imputed = data.copy()
	for i1 in range(len(ccs)):
		cc = ccs[i1]
		if len(cc) > 5:
			cc_spots = [c.name for c in cc]
			other_spots = [s for s in spots if s not in cc_spots]
			m = len(cc_spots)
			s = int(len(other_spots) / 10) # sampling is based on current cluster size
			values = np.zeros(shape=(m, data.shape[1], nExperts))
			for i2 in range(nExperts):
				np.random.seed(i2)
				sampled_spot = list(np.random.choice(other_spots, s, replace=False))
				ri_spots = cc_spots + sampled_spot
				ri_impute = rankMinImpute(data.loc[ri_spots,:])
				values[:,:,i2] = ri_impute.values[:m, :]
				print("[%.1f] Outer: %d/%d, inner: %d/%d" %(epsilon, i1+1, len(ccs), i2+1, nExperts))

			imputed.loc[cc_spots,:] = np.mean(values, axis=2)
		else:
			imputed.loc[cc_spots,:] = imputed_whole.loc[cc_spots,:]

	assert np.all(data.values[known_idx] > 0)
	imputed.values[known_idx] = data.values[known_idx]
	return imputed, member_df, f


def rankMinImpute(data):
	t1 = time()
	D = np.ravel(data.values) # flattened count matrix
	idx = np.where(D) # nonzero indices of D
	y = D[idx] # nonzero observed values
	n = np.prod(data.shape)
	err= 1E-12
	x_initial = np.zeros(np.prod(data.shape))
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
			B = np.reshape(b, data.shape)
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
	imputed = pd.DataFrame(data=X, index=data.index, columns=data.columns)
	assert not imputed.isna().any().any()
	t2 = time()
	#print("Rank minmization imputation for %d spots finished in %.1f seconds." %(data.shape[0],t2-t1))
	return imputed

def softThreshold(s, l):
	return np.multiply(np.sign(s), np.absolute(s - l))


def ep_perf(ho_data, ho_mask, meta_data, ori_data, cor_mat, cv_fold):
	eps = np.arange(0.2, 0.9, 0.1)
	ho_data_obj = Data.Data(count=ho_data, meta=meta_data, cormat=cor_mat)
	print("Start evaluating epsilon...")
	perf_dfs = []
	for ep in eps:
		ho_data_obj.update_ep(ep)
		model_data,_,_ = spImpute(ho_data_obj, nExperts=5)
		perf_df = utils.evalSlide(ori_data, ho_mask, ho_data, model_data, ep)
		perf_dfs.append(perf_df)
	perf_dfs = pd.concat(perf_dfs)
	perf_dfs = perf_dfs.sort_values("RMSE")
	perf_dfs["cv_fold"] = cv_fold
	return perf_dfs

def select_ep(original_data, meta_data, cor_mat, k=2):
	start_time = time()
	thre = 0.8
	training_data = utils.filterGene_sparsity(original_data, thre)

	if training_data.empty:
		print("No landmark genes selected, use 0.7 as default")
		return 0.7

	# generate k fold cross validation datasets
	ho_dsets, ho_masks = utils.generate_cv_masks(training_data,\
		training_data.columns.tolist(), 2)
	
	perf_dfs = []
	for fd in range(2):
		ho_data, ho_mask = ho_dsets[fd], ho_masks[fd]
		perf_df = ep_perf(ho_data, ho_mask, meta_data, training_data, cor_mat, fd)
		perf_dfs.append(perf_df)
	perf_dfs = pd.concat(perf_dfs)
	rmse_dfs = perf_dfs.loc[:, ["ModelName", "RMSE"]]
	print(rmse_dfs)
	mean_dfs = rmse_dfs.groupby("ModelName").mean()
	mean_dfs['epsilon'] = mean_dfs.index.to_numpy()
	ep = mean_dfs.loc[mean_dfs.RMSE == mean_dfs.RMSE.min(),"epsilon"].tolist()[0]
	end_time = time()
	print("Epislon %.2f is selected in %.1f seconds." %(ep, end_time - start_time))
	return ep

def main(data, select=1):
	count = data.count.copy()
	meta = data.meta.copy()
	cormat = data.cormat.copy()
	if select == 1:
		ep = select_ep(count, meta, cormat)
		data.update_ep(ep)
	return spImpute(data)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Algorithm variables.')
	parser.add_argument('-f', '--countpath', type=str,
                    help='path to the input raw count matrix,\
                    1st row is the gene name and 1st column is spot ID')
	parser.add_argument('-e', '--epislon', type=float, default=0.6,
                    help='threshold to filter low confident edges')
	parser.add_argument('-o', '--out_fn', type=str, default='none',
                    help='File path to save the imputed values.')
	parser.add_argument('-l', '--norm', type=str, default="none",
                    help='method to normalize data.')
	parser.add_argument('-s', '--select', type=int, default=1,
					help = 'whether infer epislon using holdout test or not.')
	parser.add_argument('-r', '--radius', type=int, default=2,
					help='distance radius in defining edges')
	parser.add_argument('-m', '--merge', type=int, default=5,
                                        help='distance radius in defining edges')
	args = parser.parse_args()
	count_fn = args.countpath
	epi = args.epislon
	out_fn = args.out_fn
	norm = args.norm
	select = args.select
	radius = args.radius
	merge = args.merge

	data = Data.Data(countpath=count_fn,radius=radius,
					merge=merge,norm=norm, epsilon=epi)
	imputed, member_df, figure = main(data, select=select)

	if out_fn != "none":
		imputed.to_csv(out_fn)
		fig_out = out_fn.split(".csv")[0] + ".png"
		member_out = out_fn.split(".csv")[0] + "_cluster_info.csv"
		figure.savefig(fig_out)
		member_df.to_csv(member_out)
