#!/usr/bin/env python
"""Provides ST graph related objects including Node and CC, and other
associated functions mainly to get LCNs.
"""

import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import imageio
import seaborn as sns
from shutil import rmtree
import os
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

__author__ = "Linhua Wang"
__license__ = "GNU General Public License v3.0"
__maintainer__ = "Linhua Wang"
__email__ = "linhuaw@bcm.edu"

class Node:
	"""
    A class used to represent a graph node in ST

    ...

    Attributes
    ----------
    x : int
        the horizontal coordinate of the spot
    y : int
        the vertical coordinate of the spot. NOTE: x, y can be flipped.
    name : str
        the name of the spot
    component : int
        mapping of spot to the connected components
    neighbors : list of Node
        the neighboring spots

    Methods
    -------
    isNode(node):
    	Return if node equals to current node.

	contain_neighbor(self, node):
		Return if node in the neighborhood.
	
	add_neighbor(self, node):
		Method to add a node to current node's neighborhood
	
	assign_component(self, k):
		Method to map current node to target connected component

    """

	def __init__(self,x ,y, name):
		"""
        Parameters
        ----------
        name : str
            The name of the spot
        x : int
        	the horizontal coordinate of the spot
   		y : int
        	the vertical coordinate of the spot. NOTE: x, y can be flipped.
        """
		self.x = x
		self.y = y
		self.name = name
		self.component = 0
		self.neighbors = []

	def __repr__(self):
		"""Method used to represent Node's objects as a string
		"""
		return "X: %d, Y: %d, spotID: %s, #NBs: %d" %(self.x, self.y, self.name, len(self.neighbors))

	def isNode(self, node):
		"""Method to check if two Node objects are equal

		Parameter:
		---------
			node: external Node object

		Return:
		-------
			True if two nodes' coordinates are the same
		"""
		return (self.x == node.x and self.y == node.y)

	def contain_neighbor(self, node):
		"""Method to check if external node is in current nodes' neighborhood

		Parameter:
		---------
			node: external Node object

		Return:
		-------
			True if external node is in the neighborhood; False otherwise.
		"""
		if len(self.neighbors) == 0:
			return False
		for node1 in self.neighbors:
			if node1.isNode(node):
				return True
		return False

	def add_neighbor(self, node):
		"""Method to add a node to current node's neighborhood
		
		Parameter:
		---------
			node: external Node object
		"""
		if not self.contain_neighbor(node):
			self.neighbors.append(node)

	def assign_component(self, k):
		"""Method to map current node to target connected component
		
		Parameter:
		---------
			k: int, the index of the connected component to be mapped to
		"""
		self.component=k


class CC:
	"""
    A class used to represent a connected component in ST

    ...

    Attributes
    ----------
    nodes : list of Node objects in the CC
    name : str, the name of the spot
    size : int, number of nodes in the CC

    Methods
    -------
    distance(self, node):
		Method calculating the distance between a node and the CC
	append(self, node):
		Method to add a node to current CC
    """

	def __init__(self, nodes, name):
		self.nodes = nodes
		self.name = name
		self.size = len(nodes)

	def distance(self, node):
		"""Method calculating the distance between a node and the CC

		Parameters:
		----------
		node: Node object
		"""
		dist = np.Inf
		for node2 in self.nodes:
			dist = min(dist, np.linalg.norm(np.array((node.x, node.y)) -
				np.array((node2.x, node2.y))))
		return dist

	def append(self, node):
		"""Method to add a node to current CC"""
		self.nodes.append(node)
		self.size += 1

def construct_graph(meta_data,radius=2):
	"""Method to construct the graph based on the ST meta data

	Parameters:
	----------
	meta_data: data frame, 
				index: spot ID,
				first column (x): x coordinate 
				second column (y): y coordinate
	radius: float, the radius of euclidean distance to define neighbors

	Return:
	------
	A list of nodes with neighborhood defined.
	"""
	xs, ys = meta_data.iloc[:,0].tolist(), meta_data.iloc[:,1].tolist()
	spots = meta_data.index.tolist()
	## Initialize nodes 
	nodes = []
	for i in range(len(xs)):
		nodes.append(Node(xs[i], ys[i], spots[i]))
	## Get neighbors for each node if they have distance less than radius
	for node1 in nodes:
		for node2 in nodes:
			dist = np.linalg.norm(np.array((node1.x, node1.y)) -
				np.array((node2.x, node2.y)))
			if dist < radius:
				node1.add_neighbor(node2)
				node2.add_neighbor(node1)
	return nodes

def removeNodes(nodes, cnns):
	"""Helper function for function spatialCCs.
	Method to get the remaining nodes that have not been mapped to a CC yet

	Parameters:
	----------
	nodes: list of Node not yet mapped
	cnns: current connected components

	Return:
	-------
	Remaining nodes not been mapped to a CC yet
	"""
	updated_nodes = []
	for i in range(len(nodes)):
		if nodes[i] not in cnns:
			updated_nodes.append(nodes[i])
	return updated_nodes

def spatialCCs(nodes, cor_mat, epi, merge=0):
	"""Method to get connected components by filtering out low-weight edges

	Parameters:
	----------
	nodes: list of Node in the graph
	cor_mat: spot by spot similarity matrix
	epi: the epsilon value to filter edges. 
			Edges with weights < epi will be removed.
	merge: #spots for small ccs to be merged. Discard this param for now.

	Return:
	-------
	List of list of nodes. Each list of Node will be a CC. 
	"""
	ccs = [] # initiate CCs
	while len(nodes) > 0:
		## Recursively detect CCs until no nodes remaining
		cnns = set([])
		node = nodes[0]
		## Use DFS to detect all connected nodes from current
		cnns = DFS(cnns, nodes, node, cor_mat, epi)
		ccs.append(list(cnns))
		## Remove nodes in current CC from remaining Node list
		nodes = removeNodes(nodes, cnns)

	if merge > 0:
		k = 1
		small_nodes = []
		largeCCs = []
		for cc in ccs:
			if len(cc) <= merge:
				for node in cc:
					small_nodes.append(node)
			else:
				largeCCs.append(CC(cc, k))
				k += 1

		if len(largeCCs) == 0:
			return [small_nodes]

		merge_dict = {}
		for small_node in small_nodes:
			dist = np.Inf
			idx = 0
			for i in range(len(largeCCs)):
				if largeCCs[i].distance(small_node) < dist:
					dist = largeCCs[i].distance(small_node)
					idx = i
			#largeCCs[idx].append(small_node)
			merge_dict[small_node.name] = idx
		for small_node in small_nodes:
			largeCCs[merge_dict[small_node.name]].append(small_node)
		return [largeCC.nodes for largeCC in largeCCs]
		
	return ccs


def DFS(CNNs, nodes, node, cor_mat, epi):
	"""Method to detect connected node from the current node using
	Depth First Search Algorithm developed by R Tarjan, 1972. 

	Parameters:
	----------
	CNNs: current sets of detected CNNs.
	nodes: list of remaining in the graph
	node: next node to start evaluate neighbors.
	cor_mat: spot by spot similarity matrix
	epi: the epsilon value to filter edges. 
			Edges with weights < epi will be removed.
	Return:
	-------
	sets of CNNs. 
	"""
	if node not in CNNs:
		CNNs.add(node)
		for neighbor in node.neighbors:
			if (neighbor in nodes) and \
			(cor_mat.loc[neighbor.name, node.name] >= epi):
			# adding to current CNN if is qualified neighbor
				DFS(CNNs, nodes, neighbor, cor_mat, epi)
	return CNNs

def assign_membership(ccs, meta):
	"""Method to assign spatial clustership for each spot. 

	Parameters:
	----------
	ccs: a list of CCs.
	meta: data frame, 
				index: spot ID,
				first column (x): x coordinate 
				second column (y): y coordinate
	Return: Pandas Data Frame with cluster information assigned.
	-------
	sets of CNNs. 
	"""
	cc10, cc2, cc1 = 0, 0, 0
	xmin, xmax = meta.iloc[:, 0].min(), meta.iloc[:, 0].max()
	ymin, ymax = meta.iloc[:, 1].min(), meta.iloc[:, 1].max()
	colors = ["red", "orange", "yellow", "green",
				 "cyan", "blue", "purple", "gray",
				  "pink", "black"]
	df = meta.copy()
	df["size"] = 1
	df.columns=['x','y', 'size']
	df['k'] = -1
	for i in range(len(ccs)):
		cc = ccs[i]
		for node in cc:
			x, y = node.x, node.y
			df.loc[(df.x == x) & (df.y==y),"size"] = len(cc)
			df.loc[(df.x == x) & (df.y==y),"k"] = i
	df = df.sort_values("size", ascending=False)
	df['cluster_name'] = df[['size','k']].apply(lambda x: "_".join(x.astype(str)), axis=1)
	unique_sizes = df["cluster_name"].drop_duplicates().tolist()
	cluster_dict = dict(zip(unique_sizes, 
		range(len(unique_sizes))))
	clusters = []
	for s in df["cluster_name"].tolist():
		i = cluster_dict[s]
		if i > len(colors) - 1:
			clusters.append("lightgray")
		else:
			clusters.append(colors[i])
	df['cluster'] = clusters
	return df

def detectSDEs(fn, ep, log=False):
	"""Method to detect spatially differentially expressed genes
	using Wilcoxon Rank-sum 

	Parameters:
	----------
	fn: str, path to the data.
	ep: float, edge filtering parameter.
	log: bool, whether take log of gene expression or not.

	Return: Pandas Data Frame with SDEGs pvalues and logFC.
	"""

	## Process data and get CCs
	count, meta = read_ST_data(fn)

	if log:
		count_filt = np.log2(count + 1)
	else:
		count_filt = count.copy()
	cor_mat = spot_PCA_sims(count_filt)

	nodes = construct_graph(meta)
	ccs = spatialCCs(nodes, cor_mat, ep, merge=0)

	## For each CC with >= 5 spots, conduct 1 vs others DEG analysis
	genes = count_filt.columns.tolist()
	cc_dfs = []
	for i in range(len(ccs)):
		cc = ccs[i]
		if len(cc) >= 5:
			cc_spots= [c.name for c in cc]
			count_cc = count_filt.loc[cc_spots, genes]
			other_spots = [s for s in count_filt.index.tolist() if s not in cc_spots]
			count_other = count_filt.loc[other_spots, genes]
			pvals, logFCs = [], []
			for g in genes:
				pval = ranksums(count_cc.loc[:,g].to_numpy(), count_other.loc[:, g].to_numpy())[1]
				logFC = np.mean(count_cc.loc[:,g].to_numpy()) - np.mean(count_other.loc[:, g].to_numpy())
				pvals.append(pval)
				logFCs.append(logFC)
			cc_df = pd.DataFrame({"gene":genes, "pval": pvals, "logFC":logFCs})
			cc_df["padj"] = multipletests(cc_df.pval.to_numpy())[1]
			cc_df=cc_df.loc[cc_df.padj <= 0.05,:]
			cc_df["component"] = i
			cc_dfs.append(cc_df)
	cc_dfs = pd.concat(cc_dfs)
	return cc_dfs