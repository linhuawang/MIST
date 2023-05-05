#!/usr/bin/env python
"""Imputer class create adaptors to run MIST 
	and other imputation methods
"""
import sys
import MIST2
import neighbors
import Data
import numpy as np
import pandas as pd

__author__ = "Linhua Wang"
__license__ = "GNU General Public License v3.0"
__maintainer__ = "Linhua Wang"
__email__ = "linhuaw@bcm.edu"

class Imputer(object):
	"""Object to represent one kind of imputation algorithm
	Parameters:
	----------
	name: str
			name of the imputation algorithm
	data: Data object

	Methods:
	-------
	fit_transform():
		Impute the gene expression matrix in Data object
	"""
	def __init__(self, name, data):
		assert name in ["MAGIC", "knnSmooth", "mcImpute", "spKNN", "MIST"]
		# assert isinstance(data, Data.Data)
		self.name = name
		self.data = data

	def fit_transform(self, n_neighbors=4, select=0, 
		   ncores=10, nExperts=10, thre=0, layer='CPM'):
		"""Method to call imputation methods"""
		if self.name == 'MIST':
			imputed = MIST2.MIST(self.data, nExperts=nExperts, 
				ncores=ncores, verbose=1, layer=layer)
			return imputed
		else:
			data = pd.DataFrame(data=self.data.adata.layers['CPM'].toarray(),
				index = self.data.adata.obs.index, columns = self.data.adata.var_names)
			if self.name == "MAGIC":
				return MAGIC(data)
			elif self.name == "knnSmooth":
				return knnSmooth(data, k=n_neighbors)
			elif self.name == "mcImpute":
				return mcImpute(data)
			elif self.name == "spKNN":
				corrs = self.data.adata.obsp['raw_weights'].copy()
				coord_df = self.data.adata.obs.copy()
				coord_df = coord_df[['array_col', 'array_row']]
				coord_df.columns= ["x", "y"]

				cor_df = pd.DataFrame(data=corrs,
										index=self.data.adata.obs_names.copy(),
										columns=self.data.adata.obs_names.copy())
				return spKNN(data, coord_df, cor_df, thre) 


def MAGIC(data):
	try:
		import magic
	except:
		print("MAGIC not installed. Please install before running.")
		sys.exit(0)
	""""Adaptor method to call MAGIC to impute
		For this manuscript, we used default parameters
		to call MAGIC develope dby David van Dijk, et al., 2017.

	Parameter:
	---------
	data: data frame, data to be imputed

	Return:
	------
	Imputed gene expression as data frame.
	"""
	magic_operator = magic.MAGIC(verbose=0)
	return magic_operator.fit_transform(data)

def knnSmooth(data, k=4):
	try:
		import knn_smooth
	except:
		print("KNN smooth not installed. Please install before running.")
		sys.exit(0)
	""""Adaptor method to call knnSmooth to impute
	For this manuscript, we used k=4 to make it comparable to spKNN.
	This algorithm is developed by Wagner et al., 2017.
	We adjusted the original code to make normalization consistent
	in the knnSmooth.py.

	Parameter:
	---------
	data: data frame, data to be imputed

	Return:
	------
	Imputed gene expression as data frame.
	"""
	vals = knn_smooth.knn_smoothing(np.float64(data.T.values), k).T
	return pd.DataFrame(data=vals, index=data.index, columns=data.columns)

def mcImpute(data):
	""""Adaptor method to call McImpute to impute.
	For this manuscript, we translated the MATLAB code 
	developed by (A Mongia et al., 2019) into Python.

	Parameter:
	---------
	data: data frame, data to be imputed

	Return:
	------
	Imputed gene expression as data frame.
	"""
	return MIST2.rankMinImpute([data, 'mcImpute'])

def spKNN(data, coord_df, cor_df, thre=0):
	""""A base-line method that estimate missing values by averaging
	 no more than 4 spatially adjacent spots.

	Parameter:
	---------
	data: data frame, data to be imputed
	nodes: a list of Node object
	
	Return:
	------
	Imputed gene expression as data frame.
	"""
	## construct graph
	nodes = neighbors.construct_graph(coord_df)
	print("Graph constructed.")
	nb_df = pd.DataFrame(index=data.index, columns=data.index,
				data=np.zeros((data.shape[0], data.shape[0])))

	for node in nodes:
		spot = node.name
		nbs = [nb.name for nb in node.neighbors]
		nbs = [nb for nb in nbs if cor_df.loc[spot, nb] > thre]
		nb_df.loc[spot, nbs] = 1
	print("Neighbors extracted.")
	## start impute
	count_knn = data.copy()
	for spot in data.index:
		# find neighbors for each spots
		nbs = nb_df.columns[nb_df.loc[spot,:] == 1].tolist()
		# find missing values for each spot
		missing_genes = data.columns[data.loc[spot,:] == 0]
		# impute missing values by averaging neighboring spots
		count_knn.loc[spot, missing_genes] = data.loc[nbs, missing_genes].mean(axis=0)
	return count_knn