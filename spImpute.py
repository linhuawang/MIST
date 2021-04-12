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
import datetime

def spImpute(data_obj, nExperts=10, ncores=1, plot=False, verbose=1): # multiprocessing not implemented yet
	start_time = time()
	data = data_obj.count
	meta = data_obj.meta
	epsilon = data_obj.epsilon
	radius = data_obj.radius
	nodes = data_obj.nodes
	merge = data_obj.merge
	cor_mat = data_obj.cormat
	ccs = spatialCCs(nodes, cor_mat, epsilon, merge=0)

	n1 = float(sum([len(cc) for cc in ccs if len(cc) > 5]))

	if verbose == 1:
		print("Proportion explained by connected components: %.2f" %(n1/data.shape[0]))

	if data_obj.refData == None:
		imputed_whole = rankMinImpute(data)
	else:
		imputed_whole = data.refData

	t1 = time()
	if verbose == 1:
		print("Base line imputation done in %.1f seconds ..." %(t1  - start_time))
	if plot:
		member_df = plot_ccs(ccs, meta, "epsilon = %.2f" %epsilon)

	if (len(ccs) == 1) and (plot):
		return imputed_whole, member_df
	elif (len(ccs) == 1):
		return imputed_whole

	spots = data.index.tolist()
	known_idx = np.where(data.values)
	imputed = data.copy()
	nspots = 0

	for i1 in trange(len(ccs)):
		cc = ccs[i1]
		nspots += len(cc)
		if len(cc) >= 5:
			cc_spots = [c.name for c in cc]
			other_spots = [s for s in spots if s not in cc_spots]
			m = len(cc_spots)
			s = np.min([int(len(other_spots) / 10), m]) # sampling is based on current cluster size
			values = np.zeros(shape=(m, data.shape[1], nExperts))
			
			### Get parameters for rank minimization
			if ncores > 1: # Parallel computing
				ensemble_inputs = []
				for i2 in range(nExperts):
					np.random.seed(i2)
					sampled_spot = list(np.random.choice(other_spots, s, replace=False))
					ri_spots = cc_spots + sampled_spot
					ensemble_inputs.append(data.loc[ri_spots,:])
				p = Pool(ncores)
				ensemble_outputs = p.map(rankMinImpute, ensemble_inputs)
				p.close()
			else: # Sequential computing
				ensemble_outputs = []
				for i2 in trange(nExperts):
					sampled_spot = list(np.random.choice(other_spots, s, replace=False))
					ri_spots = cc_spots + sampled_spot
					ensemble_outputs.append(rankMinImpute(data.loc[ri_spots,:]))

			# re-organize output from individual expert
			for i in range(nExperts):
				values[:,:,i] = ensemble_outputs[i].loc[cc_spots,:].values
			imputed.loc[cc_spots,:] = np.mean(values, axis=2)

		# use whole slide-imputed values 
		else:
			cc_spots = [c.name for c in cc]
			imputed.loc[cc_spots,:] = imputed_whole.loc[cc_spots,:]

	assert np.all(data.values[known_idx] > 0)
	imputed.values[known_idx] = data.values[known_idx] 

	if plot:
		return imputed, member_df
	else:
		return imputed


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
	st = time()
	eps = np.arange(0.4, 0.9, 0.1)
	ho_data_obj = Data.Data(count=ho_data, meta=meta_data, cormat=cor_mat)
	imp_whole = rankMinImpute(ho_data)
	#print("Whole slide imputation done in %d seconds." %(time() - st))
	ho_data_obj.update_refData(imp_whole)
	#print("Start evaluating epsilon...")
	perf_dfs = []
	for ep in eps:
		ho_data_obj.update_ep(ep)
		model_data = spImpute(ho_data_obj, nExperts=5, plot=False, verbose=0)
		perf_df = utils.evalSlide(ori_data, ho_mask, ho_data, model_data, ep)
		perf_dfs.append(perf_df)
	perf_dfs = pd.concat(perf_dfs)
	perf_dfs = perf_dfs.sort_values("RMSE")
	perf_dfs["cv_fold"] = cv_fold
	return perf_dfs

def select_ep(original_data, meta_data, cor_mat, k=2):
	thre = 0.8
	training_data = utils.filterGene_sparsity(original_data, thre)

	if training_data.empty:
		print("No landmark genes selected, use 0.7 as default")
		return 0.7

	# generate k fold cross validation datasets
	ho_dsets, ho_masks = utils.generate_cv_masks(training_data,\
		training_data.columns.tolist(), 2)

	perf_dfs = []

	for fd in range(1):
		ho_data, ho_mask = ho_dsets[fd], ho_masks[fd]
		perf_df = ep_perf(ho_data, ho_mask, meta_data, training_data, cor_mat, fd)
		perf_dfs.append(perf_df)

	perf_dfs = pd.concat(perf_dfs)
	rmse_dfs = perf_dfs.loc[:, ["ModelName", "RMSE"]]
	mean_dfs = rmse_dfs.groupby("ModelName").mean()
	mean_dfs['epsilon'] = mean_dfs.index.to_numpy()
	ep = mean_dfs.loc[mean_dfs.RMSE == mean_dfs.RMSE.min(),"epsilon"].tolist()[0]
	
	return ep

def main(data, ncores=1, select=1, plot=False):
	count = data.count.copy()
	meta = data.meta.copy()
	cormat = data.cormat.copy()
	start_time = time()
	ep = data.epsilon
	if select == 1:
		ep = select_ep(count, meta, cormat)
		data.update_ep(ep)	
	end_time = time()
	print("Epsilon %.2f selected in %.2f seconds." %(ep, end_time-start_time))	
	return spImpute(data, nExperts=10, ncores=ncores, plot=plot, verbose=1)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Algorithm variables.')
	parser.add_argument('-f', '--countpath', type=str,
                    help='path to the input raw count matrix,\
                    1st row is the gene name and 1st column is spot ID')
	parser.add_argument('-e', '--epislon', type=float, default=0.6,
                    help='threshold to filter low confident edges')
	parser.add_argument('-o', '--out_fn', type=str, default='none',
                    help='File path to save the imputed values.')
	parser.add_argument('-l', '--norm', type=str, default="cpm",
                    help='method to normalize data.')
	parser.add_argument('-s', '--select', type=int, default=1,
					help = 'whether infer epislon using holdout test or not.')
	parser.add_argument('-r', '--radius', type=int, default=2,
					help='distance radius in defining edges')
	parser.add_argument('-n', '--ncores', type=int, default=1,
					help='number of processors to be used in parallel computing.')
	
	args = parser.parse_args()
	count_fn = args.countpath
	epi = args.epislon
	out_fn = args.out_fn
	norm = args.norm
	select = args.select
	radius = args.radius
	ncores = args.ncores

	raw, meta = utils.read_ST_data(count_fn)
	genes = raw.columns[((raw > 2).sum() >= 2)]

	raw, libsize = utils.data_norm(raw, norm)
	impute_input = raw.loc[:, genes]

	print('Input shape:  %d spots, %d genes.' %(impute_input.shape[0], 
		impute_input.shape[1]))

	data = Data.Data(count=impute_input, meta=meta,radius=radius,
					epsilon=epi)

	imputed, member_df = main(data, ncores, select=select, plot=True)

	imputed_final = raw.copy()
	imputed_final.loc[imputed.index, imputed.columns] = imputed.values
	imputed_revNorm = utils.data_denorm(imputed_final, libsize=libsize, method=norm)

	if out_fn != "none":
		imputed.to_csv(out_fn)
		# include genes and spots that were filtered out
		complete_fn = out_fn.split(".csv")[0] + "_complete.csv"
		imputed_final.to_csv(complete_fn)
		# de-normalize to raw counts
		raw_fn = out_fn.split(".csv")[0] + "_rawCount.csv"
		imputed_revNorm.to_csv(raw_fn)
		# save cluster information
		member_out = out_fn.split(".csv")[0] + "_cluster_info.csv"
		member_df.to_csv(member_out)
