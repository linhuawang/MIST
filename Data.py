#!/usr/bin/env python
"""Provide ST Data object source code and functions to update attributes.
"""

import neighbors
import utils
import pandas as pd
import sys
sys.setrecursionlimit(10000)

__author__ = "Linhua Wang"
__license__ = "GNU General Public License v3.0"
__maintainer__ = "Linhua Wang"
__email__ = "linhuaw@bcm.edu"

class Data(object):
	"""
    A class used to represent an ST data object

    ...

    Attributes
    ----------
    count: data frame
    		gene expression, required if countpath is not provided.
    meta : data frame
        coordinates of each spot, required if countpath is not provided.
    cormat : data frame
        spot by spot similarity
    countpath : str
    		path to the gene expression file,
    		required if either count or meta is not provided.
    radius: float
        euclidean distance defining neighbor connectivity
    merge: int
    	lower limit of CC size. Not needed.
    norm: str
    	value has to be one of ['none', 'cpm', logCPM', 'logMed']
		count normalization approach to be used.
	epsilon: edge filtering parameter.

    Methods
    -------
    update_ep(ep)
    	Method to update epsilon value for Data

	update_cormat(cormat)
		Method to update spot-spot correlation for Data
	
	update_refData(refData)
		Method to update whole-slide rank minimization results for Data
	
	"""
	def __init__(self, count=None, meta=None, cormat=None,
				 countpath="", radius=2, merge=0, norm="none", epsilon=0.7,
				corr_methods=['spearman'], n_pcs=10):
		self.norm = norm
		## add count and meta data
		if countpath != "":
			self.filename = countpath
			count, meta = utils.read_ST_data(self.filename)
			count.fillna(0, inplace=True)
		else:
			assert isinstance(count, pd.DataFrame) and isinstance(meta, pd.DataFrame)

		if self.norm != "none":
			count, libsize = utils.data_norm(count, method=self.norm)
			self.libsize=libsize
			
		self.count = count
		self.meta = meta

		if isinstance(cormat, pd.DataFrame):
			self.cormat = cormat
		else:
			print("Calculating pairwise spot similarities. This step takes minutes ...")
			self.cormat = utils.spot_PCA_sims(self.count, methods=corr_methods, n_pcs=n_pcs)

		# add other features
		self.radius = radius
		self.nodes = neighbors.construct_graph(self.meta, self.radius)
		self.merge= merge
		self.epsilon=epsilon
		self.refData=None

	def update_ep(self, ep):
		"""Method to update epsilon value for Data"""
		self.epsilon=ep

	def update_cormat(self, cormat):
		"""Method to update epsilon value for spot-spot correlation"""
		self.cormat = cormat
		
	def update_refData(self, refData):
		"""Method to update the whole slide imputation result"""
		self.refData
