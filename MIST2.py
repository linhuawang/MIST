#!/usr/bin/env python
"""The main script to run MIST
"""

from neighbors import *
import pandas as pd
import numpy as np
from multiprocessing import Pool, set_start_method, get_context
from time import time
import Data
from sklearn.model_selection import KFold

__author__ = "Linhua Wang"
__license__ = "GNU General Public License v3.0"
__maintainer__ = "Linhua Wang"
__email__ = "linhuaw@bcm.edu"

def MIST(rdata, nExperts=10, ncores=1, verbose=1, layer='CPM'):
	"""
	The function that takes ST data object and return the imputed gene expression and \
	region membership for all ST spots.

	Parameters
	----------
	"""
	start_time = time()

	## Get the region assignment
	member_df = rdata.adata.obs.copy()
	isolated_spots = member_df.loc[member_df.region_ind == 'isolated', 'new_idx'].tolist()
	core_regions = set(member_df.region_ind)
	core_regions.remove("isolated")
	core_regions = list(core_regions)

	## Construct a data frame for denoising
	if layer!='CPM':
		print(f"Imputing layer {layer}")
		values = np.round(rdata.adata.X.toarray(), 2)
	else:
		print(f"Imputing layer {layer}")
		values = np.round(rdata.adata.layers['CPM'].toarray(), 2)
	data = pd.DataFrame(data=values, index = rdata.adata.obs.new_idx, columns = rdata.adata.var_names)

	spots = data.index.tolist()
	# Indices of non-zero gene expression values in the matrix
	known_idx = np.where(data.values)
	# Make template to save results
	imputed = data.copy()

	isolated_dfs = []

	nspots = 0
	# Impute each CC
	for i1 in range(len(core_regions)):
		ts = time()
		region = core_regions[i1]
		region_spots = member_df.loc[member_df.region_ind == region, 'new_idx'].tolist()
		nspots += len(region_spots)
		print(f"[Start][Region] {i1} / {len(core_regions)} | [Spots] {len(region_spots)} | {nspots} / {member_df.shape[0] - len(isolated_spots)}.")
		# QC
		# Spots from other core_regions
		other_spots = [s for s in spots if ((s not in region_spots) and (s not in isolated_spots))]

		# Randomly split isolated spots and other core region spots into nExperts splits
		kfi = KFold(n_splits=nExperts, random_state=2022-i1, shuffle=True)
		kfo = KFold(n_splits=nExperts, random_state=2022-i1, shuffle=True)

		isolated_ksplit = list(kfi.split(np.transpose(np.array(isolated_spots)[np.newaxis])))
		other_ksplit = list(kfo.split(np.transpose(np.array(other_spots)[np.newaxis])))

		### Get parameters for rank minimization
		values = np.zeros(shape=(len(region_spots), data.shape[1], nExperts))
		# Parallel computing
		ensemble_inputs = []
		for i2 in range(nExperts):
			iso_inds, other_inds = isolated_ksplit[i2][1], other_ksplit[i2][1]
			ri_spots  = region_spots + list(np.array(isolated_spots)[iso_inds]) + list(np.array(other_spots)[other_inds])
#			ri_spots  = region_spots + list(np.array(isolated_spots)[iso_inds])
			ensemble_inputs.append((data.loc[ri_spots,:], f'{i1} / {i2}'))

		with get_context("spawn").Pool(ncores) as pool:
			ensemble_outputs = pool.map(rankMinImpute, ensemble_inputs)

		ensemble_outputs = pd.concat(ensemble_outputs)
		ensemble_outputs = ensemble_outputs.groupby(ensemble_outputs.index).mean()
		imputed.loc[region_spots,:] = ensemble_outputs.loc[region_spots,imputed.columns]

		isolated_df = ensemble_outputs.loc[ensemble_outputs.index.isin(isolated_spots),:]
		if not isolated_df.empty:
			isolated_dfs.append(isolated_df)

		te = time()
		elapsed= te - ts
		print(f"[End][Region] {i1} / {len(core_regions)} | [Elapsed] {elapsed:.1f} seconds.")

	## Get ensemble output for isolated spots
	isolated_dfs = pd.concat(isolated_dfs)
	isolated_dfs = isolated_dfs.groupby(isolated_dfs.index).mean()

	imputed.loc[isolated_dfs.index, :] = isolated_dfs.loc[:, imputed.columns].values

	## Make sure original non-zero values are not perturbed
	imputed.values[known_idx] = data.values[known_idx] 
	return imputed

def rankMinImpute(param):
	"""
	The function that impute each LCN. Modified based on the\
	 MATLAB code from Mongia, A., et. al., (2019).

	Input: expression matrix with rows as samples and columns as genes
	Output: imputed expression matrix of the same size.
	"""
	data, strFlag = param
	print(f'[{strFlag}] imputation started with {data.shape[0]} observations and {data.shape[1]} varaibles.')
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
	print(f'[{strFlag}] imputation completed in {(t2-t1):.1f} seconds.')
	return imputed

def softThreshold(s, l):
	"""
	Helper function to compute the L1 gradient for function rankMinImpute.
	"""
	return np.multiply(np.sign(s), np.absolute(s - l))
