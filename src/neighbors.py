import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import imageio
from utils import *
import seaborn as sns
from shutil import rmtree
import os
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

class Node:
	def __init__(self,x ,y, name):
		self.x = x
		self.y = y
		self.name = name
		self.component = 0
		self.neighbors = []

	def __repr__(self):
		return "X: %d, Y: %d, spotID: %s, #NBs: %d" %(self.x, self.y, self.name, len(self.neighbors))

	def isNode(self, node):
		return (self.x == node.x and self.y == node.y)

	def contain_neighbor(self, node):
		if len(self.neighbors) == 0:
			return False
		for node1 in self.neighbors:
			if node1.isNode(node):
				return True
		return False

	def add_neighbor(self, node):
		if not self.contain_neighbor(node):
			self.neighbors.append(node)

	def assign_component(self, k):
		self.component=k


class CC:
	def __init__(self, nodes, name):
		self.nodes = nodes
		self.name = name
		self.size = len(nodes)

	def distance(self, node):
		dist = np.Inf
		for node2 in self.nodes:
			dist = min(dist, np.linalg.norm(np.array((node.x, node.y)) -
				np.array((node2.x, node2.y))))
		return dist

	def append(self, node):
		self.nodes.append(node)
		self.size += 1

def construct_graph(meta_data,radius=2):
	xs, ys = meta_data.iloc[:,0].tolist(), meta_data.iloc[:,1].tolist()
	spots = meta_data.index.tolist()
	nodes = []
	for i in range(len(xs)):
		nodes.append(Node(xs[i], ys[i], spots[i]))

	for node1 in nodes:
		for node2 in nodes:
			dist = np.linalg.norm(np.array((node1.x, node1.y)) -
				np.array((node2.x, node2.y)))
			if dist < radius:
				node1.add_neighbor(node2)
				node2.add_neighbor(node1)
	return nodes

def removeNodes(nodes, cnns):
	updated_nodes = []
	for i in range(len(nodes)):
		if nodes[i] not in cnns:
			updated_nodes.append(nodes[i])
	return updated_nodes

def spatialCCs(nodes, cor_mat, epi=0, merge=5):
	ccs = []
	while len(nodes) > 0:
		cnns = set([])
		node = nodes[0]
		cnns = DFS(cnns, nodes, node, cor_mat, epi)
		ccs.append(list(cnns))
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

def isValidNode(node, CNNs, cor_mat, epi):
	isValid = True
	for node2 in CNNs:
		if cor_mat.loc[node.name, node2.name] < epi:
			isValid = False
	return isValid

def DFS(CNNs, nodes, node, cor_mat, epi):
	if node not in CNNs:
		CNNs.add(node)
		for neighbor in node.neighbors:
			if (neighbor in nodes) and \
			(cor_mat.loc[neighbor.name, node.name] >= epi):
				DFS(CNNs, nodes, neighbor, cor_mat, epi)
	return CNNs

def plot_ccs(ccs, meta, title="none"):
	cc10, cc2, cc1 = 0, 0, 0
	xmin, xmax = meta.iloc[:, 0].min(), meta.iloc[:, 0].max()
	ymin, ymax = meta.iloc[:, 1].min(), meta.iloc[:, 1].max()
	colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
				'#911eb4', '#46f0f0', '#f032e6','#bcf60c', '#fabebe',
				'#008080', '#e6beff','#9a6324', '#fffac8', '#800000',
				 '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
				  '#ffffff', '#000000']
	df = meta.copy()
	df['color'] = 'black'
	df.columns=['x','y', 'color']

	for i in range(len(ccs)):
		cc = ccs[i]
		if len(cc) > 5:
			for node in cc:
				x, y = node.x, node.y
				df.loc[(df.x == x) & (df.y==y),"color"] = colors[i]
	f = plt.figure(figsize=(4,4))
	plt.scatter(x=df.x.to_numpy(), y=df.y.to_numpy(), c=df.color.tolist())
	plt.gca().invert_yaxis()
	if title != "none":
		plt.title(title)
	return f

# input count matrix should be log scaled
def detectSDEs(fn, ep, log=False):
	count, meta = read_ST_data(fn)

	if log:
		count_filt = np.log2(count + 1)
	else:
		count_filt = count.copy()
	cor_mat = spot_PCA_sims(count_filt)

	nodes = construct_graph(meta)
	ccs = spatialCCs(nodes, cor_mat, ep, merge=0)

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
	print(cc_dfs)
	return cc_dfs

if __name__ == "__main__":
	# #plot_connected_neighbors("B06_E1__17_14", cnns4, meta_data,"")
	# data, meta = read_ST_data("/Users/linhuaw/Documents/STICK/results/mouse_wt/logCPM.csv")
	fn = sys.argv[1]
	data, meta = read_ST_data(fn)
#	data, meta = read_ST_data("/Users/linhuaw/Documents/STICK/analysis/Holdout Experiments/data/MouseWT/ho_data/ho_data_0.csv")
	cor_mat = spot_PCA_sims(data)
	nodes = construct_graph(meta)
	for e in [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
		ccs = spatialCCs(nodes, cor_mat, e)
		print(len(ccs))
		plot_ccs(ccs, meta)
