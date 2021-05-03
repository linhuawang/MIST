#!/usr/bin/env python
"""The main script to run MIST
"""

import utils
from neighbors import *
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
from shutil import rmtree
from scipy.sparse import csgraph
from tqdm import tqdm, trange
import Data
import datetime

__author__ = "Linhua Wang"
__license__ = "GNU General Public License v3.0"
__maintainer__ = "Linhua Wang"
__email__ = "linhuaw@bcm.edu"

def MIST(data_obj, nExperts=10, ncores=1, verbose=1):
	"""
	The function that takes ST data object and return the imputed gene expression and \
	region membership for all ST spots.

	Parameters
	----------
	data_obj: Data object that contains:
				* data_obj.count:	a normalized count matrix as
				* data_obj.meta: coordinates of each spot
				* data_obj.epsilon: fixed epsilon value
				* data_obj.nodes: a defined graph with nodes
				* data_obj.cormat: the PCA distance similarity of paired spots

	nExperts: #runs of rank minimization with augmented connected components to average
	ncores: #processors to be used for parallel computing. Range should be in 1~#Experts.
	verbose: wheter or not to print the intermediate info. 1 for yes and 0 otherwise.
	"""

	start_time = time()
	# Get parameters
	data = data_obj.count
	meta = data_obj.meta
	epsilon = data_obj.epsilon
	radius = data_obj.radius
	nodes = data_obj.nodes
	cor_mat = data_obj.cormat

	# Get the connected components (CC) using DFS algorithm
	ccs = spatialCCs(nodes, cor_mat, epsilon, merge=0)
	# Get the number of CCs that are qualified for rank minimization
	n1 = float(sum([len(cc) for cc in ccs if len(cc) > 5]))
	if verbose == 1:
		print("Proportion explained by connected components: %.2f" %(n1/data.shape[0]))

	# Get the rank minimization results using all spots
	if data_obj.refData == None:
		imputed_whole = rankMinImpute(data)
	else:
		imputed_whole = data.refData

	t1 = time()
	if verbose == 1:
		print("Base line imputation done in %.1f seconds ..." %(t1  - start_time))
	
	# Get the spot-CC assignment map
	member_df = assign_membership(ccs, meta)
	# Return whole-slide imputation if no local regions detected
	if (len(ccs) == 1):
		return imputed_whole, member_df

	spots = data.index.tolist()
	# Indices of non-zero gene expression values in the matrix
	known_idx = np.where(data.values)
	# Make template to save results
	imputed = data.copy()
	nspots = 0
	# Impute each CC
	for i1 in trange(len(ccs)):
		cc = ccs[i1]
		nspots += len(cc)
		# QC
		if len(cc) >= 5:
			# Get LCNs
			cc_spots = [c.name for c in cc]
			# Augment LCNs with random spots
			other_spots = [s for s in spots if s not in cc_spots]
			m = len(cc_spots)
			s = np.min([int(len(other_spots) / 10), m]) # sampling is based on current cluster size
			# Matrix to save multiple runs of rank minimization
			values = np.zeros(shape=(m, data.shape[1], nExperts))
			
			### Get parameters for rank minimization
			# Parallel computing
			if ncores > 1: 
				ensemble_inputs = []
				for i2 in range(nExperts):
					np.random.seed(i2)
					sampled_spot = list(np.random.choice(other_spots, s, replace=False))
					ri_spots = cc_spots + sampled_spot
					ensemble_inputs.append(data.loc[ri_spots,:])
				p = Pool(ncores)
				ensemble_outputs = p.map(rankMinImpute, ensemble_inputs)
				p.close()
			else: 
				# Sequential computing
				ensemble_outputs = []
				for i2 in trange(nExperts):
					sampled_spot = list(np.random.choice(other_spots, s, replace=False))
					ri_spots = cc_spots + sampled_spot
					ensemble_outputs.append(rankMinImpute(data.loc[ri_spots,:]))

			# Reorganize and average multiple runs of rank minimization results for CC
			for i in range(nExperts):
				values[:,:,i] = ensemble_outputs[i].loc[cc_spots,:].values
			imputed.loc[cc_spots,:] = np.mean(values, axis=2) 
		else:
			# For CCs that didn't pass QC, use whole-slide imputation results
			cc_spots = [c.name for c in cc]
			imputed.loc[cc_spots,:] = imputed_whole.loc[cc_spots,:]
	## Make sure original non-zero values are not perturbed
	imputed.values[known_idx] = data.values[known_idx] 
	return imputed, member_df

def rankMinImpute(data):
	"""
	The function that impute each LCN. Modified based on the\
	 MATLAB code from Mongia, A., et. al., (2019).

	Input: expression matrix with rows as samples and columns as genes
	Output: imputed expression matrix of the same size.
	"""
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
	imputed = pd.DataFrame(data=X, index=data.index, columns=data.columns)
	assert not imputed.isna().any().any()
	t2 = time()
	return imputed

def softThreshold(s, l):
	"""
	Helper function to compute the L1 gradient for function rankMinImpute.
	"""
	return np.multiply(np.sign(s), np.absolute(s - l))


def ep_perf(ho_data, ho_mask, meta_data, ori_data, cor_mat, cv_fold):
	"""
	Method to evaluate epsilon determined imputation results.
	"""
	st = time()
	eps = np.arange(0.4, 0.9, 0.1)# initiate epsilons to be evaluated
	# initiate the holdout data object
	ho_data_obj = Data.Data(count=ho_data, meta=meta_data, cormat=cor_mat)
	# initiate the whole matrix rank minimzation imputed results for small CCs
	imp_whole = rankMinImpute(ho_data)
	# update to the holdout object to save computational time
	ho_data_obj.update_refData(imp_whole)

	# save epsilon determined imputation performance 
	perf_dfs = []
	for ep in eps:
		ho_data_obj.update_ep(ep) # update epsilon
		model_data,_ = MIST(ho_data_obj, nExperts=5, verbose=0) # impute
		perf_df = utils.evalSlide(ori_data, ho_mask, ho_data, model_data, ep) # evaluate
		perf_dfs.append(perf_df)
	perf_dfs = pd.concat(perf_dfs)
	perf_dfs = perf_dfs.sort_values("RMSE")
	perf_dfs["cv_fold"] = cv_fold
	return perf_dfs

def select_ep(original_data, meta_data, cor_mat, thre=0.8, k=5):
	"""
	Method to automatically select optimal epsilon. 

	Parameters:
	----------
	original_data: the normalized gene expression data
	meta_data: the coordinates of each spot in a pandas data frame
	cor_mat: the spot by spot similarity matrix
	k (optional): number of fold in parameter selection
	thre (optional): parameter to define the high density genes.
	"""

	# Select high density genes for parameter learning
	training_data = utils.filterGene_sparsity(original_data, thre)
	if training_data.empty:
		# If no such genes exist, use 0.7 as default
		print("No landmark genes selected, use 0.7 as default")
		return 0.7

	# generate k fold cross validation datasets
	ho_dsets, ho_masks = utils.generate_cv_masks(training_data,\
		training_data.columns.tolist(), k)

	# In a 5 fold cross-validation fashion, select epsilon values
	# using the defined high-density genes
	perf_dfs = []
	for fd in range(k):
		ho_data, ho_mask = ho_dsets[fd], ho_masks[fd]
		perf_df = ep_perf(ho_data, ho_mask, meta_data, training_data, cor_mat, fd)
		perf_dfs.append(perf_df)
	perf_dfs = pd.concat(perf_dfs)
	rmse_dfs = perf_dfs.loc[:, ["ModelName", "RMSE"]]
	# select epsilon based on lowest average RMSE
	mean_dfs = rmse_dfs.groupby("ModelName").mean()
	mean_dfs['epsilon'] = mean_dfs.index.to_numpy()
	ep = mean_dfs.loc[mean_dfs.RMSE == mean_dfs.RMSE.min(),"epsilon"].tolist()[0]
	return ep

def main(data, ncores=1, select=1):
	"""
	Main method to run MIST pipeline.

	Prameters:
	---------
	data_obj: Data object that contains:
				* data_obj.count:	a normalized count matrix as
				* data_obj.meta: coordinates of each spot
				* data_obj.cormat: the PCA distance similarity of paired spots
	ncores: #processors to be used for parallel computing. Range should be in 1~#Experts.
	select: whether automatically select epsilon or not. 
			If not, need to have fixed epsilon already defined in data object.

	Return:
	------
		1. Imputed gene expression values
		2. Membership mapping from spots to each connected components.
	"""
	count = data.count.copy()
	meta = data.meta.copy()
	cormat = data.cormat.copy()
	start_time = time()
	ep = data.epsilon
	if select == 1:
		# Automatically select epsilon value
		ep = select_ep(count, meta, cormat)
		data.update_ep(ep)	
	end_time = time()
	print("Epsilon %.2f selected in %.2f seconds." %(ep, end_time-start_time))	
	return MIST(data, nExperts=10, ncores=ncores, verbose=1)

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

	# Parse arguments
	args = parser.parse_args()
	count_fn = args.countpath
	epi = args.epislon
	out_fn = args.out_fn
	norm = args.norm
	select = args.select
	radius = args.radius
	ncores = args.ncores
	# Read raw data
	raw, meta = utils.read_ST_data(count_fn)
	# QC by removing sparse genes
	genes = raw.columns[((raw > 2).sum() >= 2)]
	# Normalize and memorize the library size of each spot
	raw, libsize = utils.data_norm(raw, norm)
	impute_input = raw.loc[:, genes]

	print('Input shape:  %d spots, %d genes.' %(impute_input.shape[0], 
		impute_input.shape[1]))

	# Form the Data object with post-QC normalized gene expression value, 
	#	the coordinate meta data frame, the radius defining connectivity,
	#   and the prior epsilon. 
	data = Data.Data(count=impute_input, meta=meta,radius=radius,
					epsilon=epi)

	## RUN MIST
	imputed, member_df = main(data, ncores, select=select)
	## Imputed, normalized gene expression matrix with the pre-QC genes backed up
	imputed_final = raw.copy()
	imputed_final.loc[imputed.index, imputed.columns] = imputed.values
	## Imputed, raw count matrix 
	imputed_revNorm = utils.data_denorm(imputed_final, libsize=libsize, method=norm)

	if out_fn != "none":
		# save normalized gene expression
		imputed.to_csv(out_fn)
		# save normalized gene expression with pre-QC genes
		complete_fn = out_fn.split(".csv")[0] + "_complete.csv"
		imputed_final.to_csv(complete_fn)
		# save de-normalized, imputed raw counts
		raw_fn = out_fn.split(".csv")[0] + "_rawCount.csv"
		imputed_revNorm.to_csv(raw_fn)
		# save membership mapping
		member_out = out_fn.split(".csv")[0] + "_cluster_info.csv"
		member_df.to_csv(member_out)
