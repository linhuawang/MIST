import magic
import knn_smooth
import sys
sys.path.append("../src/")
#from spImpute import rankMinImpute
import spImpute
import neighbors
from Data import Data
import numpy as np
import pandas as pd
## data: n samples by p features
## all data should be normalized
class Imputer(object):
	"""docstring for Imputer"""
	def __init__(self, name, data):
		assert name in ["MAGIC", "knnSmooth", "mcImpute", "spKNN", "spImpute"]
		assert isinstance(data, Data)
		self.name = name
		self.data = data

	def fit_transform(self):
		if self.name == "MAGIC":
			return MAGIC(self.data.count)
		elif self.name == "knnSmooth":
			return knnSmooth(self.data.count, k=4)
		elif self.name == "mcImpute":
			return mcImpute(self.data.count)
		elif self.name == "spKNN":
			return spKNN(self.data.count, self.data.nodes)
		else:
			imputed, _ = spImpute.spImpute(self.data) # fixed epsilon
			return imputed

def MAGIC(data):
	magic_operator = magic.MAGIC(verbose=0)
	return magic_operator.fit_transform(data)

def knnSmooth(data, k=4):
	vals = knn_smooth.knn_smoothing(np.float64(data.T.values), k).T
	return pd.DataFrame(data=vals, index=data.index, columns=data.columns)

def mcImpute(data):
	return spImpute.rankMinImpute(data)

def spKNN(data, nodes):
	## construct graph
	nb_df = pd.DataFrame(index=data.index, columns=data.index,
				data=np.zeros((data.shape[0], data.shape[0])))
	for node in nodes:
		spot = node.name
		nbs = [nb.name for nb in node.neighbors]
		nb_df.loc[spot, nbs] = 1
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
