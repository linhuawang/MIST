from detect_regions import *
import scanpy as sc
import anndata as ad
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import sys
from statsmodels.stats.multitest import multipletests
from scipy.stats import ranksums
import warnings
from scipy.sparse import csr_matrix
from imputers import Imputer
import shutil
import os
from alphashape import alphashape
from descartes import PolygonPatch
import joblib
import plotly.express as px
import gseapy as gp
from utils import weighted_PCA_sims

class ReST(object):
	""" An Region-based Spatial Transcriptomics (ReST) object dedicated to perform 
		multiple region based algorithms and functions.
	"""
	def __init__(self, path=None, adata=None, 
		counts=None, coordinates=None, gene_df=None):
		"""Build the object 

		Parameters (one group of the following parameters):
		---------------------------------------------------
			1. path: to the folder contains the results from Space Ranger
				- spatial (folder)
				- filtered expression matrix in .h5 format
			2. adata: processed AnnData object with count, spot and gene meta data frame
			3. 
				i) counts: gene expression data frame in Pandas.DataFrame format
				ii) coordinates: spot meta data frame, with x, y columns denoting coordinates
				iii) gene_df: gene meta data frame
			4. if no parameters are given, please use load() function to load data later.

		Return:
		-------
			ReST object
		"""
		if path != None:
			adata = sc.read_visium(path)
			self.shape = adata.shape
		elif (counts is not None) and (coordinates is not None) and (gene_df is not None):
			adata = ad.AnnData(X=csr_matrix(counts.values), obs=coordinates, var=gene_df)
			self.shape = adata.shape
		elif adata is not None:
			adata.X = csr_matrix(adata.X)
			adata = adata
			self.shape = adata.shape
		else:
			adata = None
			self.shape = None
			print("Please use load() function to load data. Otherwise, no useful info. can be used.")
		self.adata = adata
		self.nodes = None
		self.species = None
		self.region_vAll_marker_dict = None
		self.auto_region_names = None
		self.region_enrichment_result = None
		self.region_color_dict = None
		self.annot_adata = None

	def shallow_copy(self):
		"""helper function to copy the object to avoid overwritten"""
		rd2 = ReST(adata=self.adata.copy())
		return rd2

	def preprocess(self, hvg_prop=0.9,species='Human', n_pcs=10, min_read_count=0, min_cell_count=0, corr_methods = ['spearman']):
		"""Important function to preprocess the data by normalization and filtering
		
		Parameters:
		-----------
		hvg_prop: float between 0 and 1, fraction of highly variable genes to be used in PCA
		species: str, either Mouse or Human, species of the sample
		n_pcs: int, number of principal components to be used when calculating the similarity matrix
		filter_spot: bool, wheter to apply spot filtering or not

		Procedures:
		-----------
		1. Filter spots with less than 1500 UMIs and more than 20% mitochondria genes
		2. Filter genes that are expressed in less than 10 spots
		3. Normalization of raw counts into logCPM to account for library size difference and smooth variance
		4. Apply PCA using highly variable genes
		5. Calculate paired similarity matrix
		"""
		adata = self.adata.copy()
		adata.var_names_make_unique()
		adata.obs['new_idx'] = adata.obs[['array_col', 'array_row']].apply(lambda x: 'x'.join(x.astype(str)), axis=1)

		print(f'Before QC: {adata.shape[0]} observations and {adata.shape[1]} genes.')

		# Procedure 1: Filter spots
		if species == 'Human':
			adata.var["mt"] = adata.var_names.str.startswith("MT-")
		elif species == 'Mouse':
			adata.var["mt"] = adata.var_names.str.startswith("mt-")
		else:
			print("Please pass a valid species name from ['Human', 'Mouse'].")
		self.species = species

		sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
		#counts = adata.obs.total_counts.to_numpy()
		
		print(f'Filtering spots with less than {min_read_count} UMIs.')
		sc.pp.filter_cells(adata, min_counts=min_read_count)
		adata = adata[adata.obs["pct_counts_mt"] < 25]
		# Procedure 2: Filter genes
		sc.pp.filter_genes(adata, min_cells=min_cell_count)
		print(f'Filtering by genes resulted in {adata.shape[1]} genes.' )
		if adata.shape[1] > 10000:
			if isinstance(adata.X, csr_matrix):
				X = adata.X.toarray()
			else:
				X = adata.X.copy()
			adata = adata[:, np.sum(X > 1, axis=0) > 1]
			del X
		print(f'After QC: {adata.shape[0]} observations and {adata.shape[1]} genes.')

		# Procedure 3: Data normalization 
		adata.raw = adata
		sc.pp.normalize_total(adata, inplace=True, target_sum=10**6)
		adata.layers['CPM'] = adata.X.copy()
		sc.pp.log1p(adata, base=2)

		# Procedure 4: Apply PCA using highly variable genes
		if hvg_prop is not None:
			sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=int(hvg_prop * adata.shape[1]))
		else:
			sc.pp.highly_variable_genes(adata)
		
		# Procedure 5: Calculate paired similarity matrix
		if corr_methods == ['weighted']:
			## this method calculates weighted similaries based on the PC-explained variance
			corrs, pca_res = weighted_PCA_sims(adata[:, adata.var.highly_variable].X.toarray(), n_pcs)
			adata.obsp['raw_weights'] = corrs
			adata.obsm['X_pca'] = pca_res
		else:	# should not contain 'weighted' in the method list if len(corr_methods) > 1
			sc.pp.scale(adata)
			sc.pp.pca(adata, n_comps=n_pcs)			
			pca_df = pd.DataFrame(data=adata.obsm['X_pca'], index=adata.obs_names)
			adata.obsp['raw_weights'] = np.mean([pca_df.T.corr(method=method).values for method in corr_methods],axis=0)
		self.adata=adata

	def extract_regions(self, min_sim=0.8, max_sim=0.96, n_pcs=10, 
					 gap=0.05, min_size=40, sigma=0.5, region_min=3, radius=2):
		"""Extract core regions by mathemetical optimization, edge pruning and modularity detection

		Parameters:
		-----------
		min_sim: float, range from 0 to 1, determines the starting point to search for the optimal threshold.
		max_sim: float, range from 0 to 1, determines the end point to search for the optimal threshold.
		gap: float, step size to in the grid search
		sigma: threshold parameter for the least acceptible isolated spots' proportions, default 0.5
		region_min: int, minimum of number of regions to search in the graph

		Procedures:
		-----------
		1. Create a MIST object
		2. Optimize the threshold
		3. Prune the graph and detect the connected components as regions
		4. Result integration
		"""

		assert min_sim < max_sim
		warnings.filterwarnings('ignore')

		# 1.Create a MIST object
		adata = self.adata.copy()
		mixture_meta = pd.DataFrame({'x':adata.obs.array_col.tolist(),
									 'y':adata.obs.array_row.tolist()}, 
										index=adata.obs['new_idx'])

		count_df = pd.DataFrame(data=adata[:, adata.var.highly_variable].X.toarray(), 
						  index=adata.obs.new_idx.tolist(), 
						 columns=adata.var.index[adata.var.highly_variable])

		cor_df = pd.DataFrame(data=adata.obsp['raw_weights'], 
						  index=adata.obs.new_idx.tolist(), 
						 columns=adata.obs.new_idx.tolist())
		t11 = time()
		count_data = Data(count=count_df, meta=mixture_meta, radius=radius, n_pcs=n_pcs, cormat=cor_df)
		t2 = time()
		print(f"MIST Data created in {(t2-t11):.2f} seconds.")

		# 2 & 3. Optimize the threshold and detect regions
		results = select_epsilon(count_data, min_sim=min_sim, 
					max_sim=max_sim, gap=gap, 
					min_region=min_size, 
					sigma=sigma, region_min=region_min)
		# 4. Result integration
		self.thr_opt_fig = results['thre_figure']
		sample_regions = results['region_df']
		count_data.epsilon = results['threshold']
		self.epsilon = results['threshold']
		self.nodes = count_data.nodes
		region_dict = dict(zip(sample_regions.index.tolist(),
							 sample_regions.region_ind))
		region_inds = []
	
		for idx in adata.obs.new_idx:
			if idx not in sample_regions.index:
				region_inds.append('isolated')
			else:
				region_inds.append(str(region_dict[idx]))
			
		adata.obs['region_ind'] = region_inds
		self.adata=adata

	def assign_region_colors(self, region_colors = None):
		""" Assign a color to each detected region. If region_colors provided, do as it
			instructed. Otherwise, use automatically assigned colors.

		@limit: <= 20 regions

		Parameters:
		-----------
		region_colors: a dictionary object if provided

		"""
		regions = set(self.adata.obs.region_ind)
		if 'isolated' in regions:
			regions.remove("isolated")

		if region_colors is None:
			colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
						'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
						'dimgrey', 'silver', 'rosybrown', 'firebrick', 'mistyrose',
							'tan', 'yellowgreen', 'dodgerblue', 'magenta', 'deepskyblue']
			if len(regions) > 20:
				colors = ['black'] * len(regions)
				print("Too many regions detected, please increase threshold for parameter min_size.")
				sys.exit(0)

			colors = colors[:len(regions)]
			region_colors = dict(zip(regions, colors))
		region_colors['isolated'] = 'gray'
		self.region_color_dict = region_colors


	def extract_regional_markers(self, mode='all'):
		"""Extract markers for each detected region using Wilcoxon rank-sum test
		"""
		adata = self.adata.copy()
		regions = set(adata.obs.region_ind)
		if 'isolated' in regions:
			regions.remove('isolated')
		regions = list(regions)
		genes = adata.var_names.tolist()

		assert mode in ['mutual', 'all']
		if mode == 'mutual':
			dfs = []
			for j in trange(len(regions) - 1):
				for k in range(j+1, len(regions)):
					reg1, reg2 = regions[j], regions[k]
					reg1_expr = adata[adata.obs.region_ind == reg1, :].layers['CPM'].toarray()
					reg2_expr = adata[adata.obs.region_ind == reg2, :].layers['CPM'].toarray()
					reg1_mean = np.mean(reg1_expr, axis=0)
					reg2_mean = np.mean(reg2_expr, axis=0)
					genes_1_2_lfcs = np.log2(np.divide(reg1_mean + 1, reg2_mean + 1))

					pvals = []
					for i in range(len(genes)):
						gex1 = np.ravel(reg1_expr[:, i])
						gex2 = np.ravel(reg2_expr[:, i])
						pval = ranksums(gex1, gex2)[1]
						pvals.append(pval)
					
					padjs = multipletests(pvals)[1]
					
					df = pd.DataFrame({'region1': reg1, 'region2': reg2, 'gene': genes,
										'lfc': genes_1_2_lfcs, 'pval': pvals, 'padj': padjs})
					df = df.loc[(np.absolute(df.lfc) > 0.26) & (df.padj <= 0.01)]
					dfs.append(df)
			dfs = pd.concat(dfs)
			dfs_pos = dfs.loc[dfs.lfc >= 0]
			dfs_neg = dfs.loc[dfs.lfc <0 ]
			dfs_neg.rename({'region1': 'region2', 'region2': 'region1'})
			dfs_neg['lfc'] = np.absolute(dfs_neg['lfc'])
			dfs_neg = dfs_neg[dfs_pos.columns]
			dfs_new = pd.concat([dfs_pos, dfs_neg])
			self.mutual_region_marker = dfs_new
		else:
			adata_cores = adata[(adata.obs.region_ind != 'isolated'),:]
			region_marker_dict = {}
			all_region_markers  = []
			for j in trange(len(regions)):
				region = regions[j]
				region_expr = adata_cores[adata_cores.obs.region_ind == region, genes].layers['CPM'].toarray()
				other_expr = adata_cores[(adata_cores.obs.region_ind != region), genes].layers['CPM'].toarray()
				region_mean = np.mean(region_expr, axis=0)
				other_mean = np.mean(other_expr, axis=0)
				lfcs = np.log2(np.divide(region_mean + 1, other_mean + 1))
				pvals = []
				for i in range(len(genes)):
					gex1 = np.ravel(region_expr[:, i])
					gex2 = np.ravel(other_expr[:, i])
					pval = ranksums(gex1, gex2)[1]
					pvals.append(pval)
				padjs = multipletests(pvals)[1]

				df = pd.DataFrame({'region_ind': region, 'region': self.region_color_dict[region],
				 'gene': genes,'lfc': lfcs, 'pval': pvals, 'padj': padjs})

				all_region_markers.append(df)
				df = df.loc[(df.lfc > 0.26) & (df.padj <= 0.01)]
				df = df.sort_values("lfc", ascending=False)
				region_marker_dict[self.region_color_dict[region]] = df

			self.region_vAll_marker_dict = region_marker_dict
			all_region_markers = pd.concat(all_region_markers)
			self.region_deg_results = all_region_markers

	def runGSEA(self, mode='all', gene_sets=None, species='Human'):
		"""Run Gene Set Enrichment Analysis using gseapy:
			https://gseapy.readthedocs.io/en/latest/introduction.html

		Parameters:
		----------
		gene_sets: list, gene sets to run the analysis
		spcies: str, either 'Mouse' or 'Human'
		"""

		print(f"Running GSEA on mode {mode} for species {species}.")

		if mode not in ['all', 'mutual']:
			print("Either mode 'all' or 'mutual' is accepted.")
			sys.exit(0)

		if gene_sets is None:
			gene_sets = ['GO_Biological_Process_2021', 'GO_Molecular_Function_2021',
			 'GO_Cellular_Component_2021',  f'KEGG_2019_{species}']

		if species not in ['Mouse', 'Human']:
			print(f"GSEA not supported for species {species}. Only 'Human' or 'Mouse' is accepted.")
			return

		region_inds = set(self.adata.obs.region_ind)
		if 'isolated' in region_inds:
			region_inds.remove("isolated")
		region_inds = list(region_inds)
		region_names = {'isolated': 'Others'}
		region_gsea_dict = {}
		for region_ind in region_inds:
			region = self.region_color_dict[region_ind]

			if mode == 'all': # use pre-rank
				region_marker_df = self.region_vAll_marker_dict[region]
				region_marker_df = region_marker_df.sort_values("lfc", ascending=False)
				region_marker_df = region_marker_df[['gene', 'lfc']]
				region_markers = list(region_marker_df.gene)
			else:
				## Currently disabled
				region_marker_df = self.region_marker_df.copy()
				region_marker_df = region_marker_df.loc[region_marker_df.region1 == region]

				marker_grps = region_marker_df.groupby("region2")
				region_markers = set(region_marker_df.gene)
				for _, mgp in marker_grps:
					region_markers = region_markers.intersection(set(mgp.gene))
				region_markers = list(region_markers)
			try:
				res = gp.enrichr(gene_list= region_markers, 
							gene_sets=gene_sets,
							organism=species)
			except:
				try:
					res = gp.enrichr(gene_list= region_markers, gene_sets=gene_sets)
				except:
					res = None
			if res is not None:
				res = res.res2d
				res = res.sort_values("Odds Ratio")
				res_sig = res.loc[res['Adjusted P-value'] <= 0.05]
				if not res_sig.empty:
					assigned_name = f"{res_sig.Term.tolist()[0]} ({region_markers[0]})"
					region_gsea_dict[region] = res_sig
				else:
					assigned_name = ', '.join(region_markers[0: np.min([3, len(region_markers)])])
			else:
				assigned_name = ', '.join(region_markers[0: np.min([3, len(region_markers)])])


			region_names[region] = assigned_name

		self.auto_region_names = region_names
		self.region_enrichment_result = region_gsea_dict
		if os.path.exists("Enrichr"):
			shutil.rmtree("Enrichr")

	def plot_regions(self, marker_size=50, mode='spot_level'):
		""" Plot spot-level region assignments via plotly
		"""
		n = int(np.sqrt(self.adata.shape[0])) / 2
		f = plt.figure(figsize=(n,n))
		if mode == 'spot_level':
			df = self.adata.obs
			if len(set(df.region_ind)) < 10:
				cmap = 'tab10'
			elif len(set(df.region_ind)) < 20:
				cmap = 'tab20'
			else:
				cmap = None
			df['region'] = df.region_ind.map(self.auto_region_names)
			term1, term2, term3, term4, term5 = [], [], [], [], []
			for region in df.region_ind.tolist():
				if region == 'isolated':
					term1.append('NA')
					term2.append('NA')
					term3.append('NA')
					term4.append('NA')
					term5.append('NA')
				else:
					terms = self.region_enrichment_result[region].sort_values("Odds Ratio", ascending=False).Term[:5].tolist()
					term1.append(terms[0])
					term2.append(terms[1])
					term3.append(terms[2])
					term4.append(terms[3])
					term5.append(terms[4])
			df['term1'] =term1
			df['term2'] =term2
			df['term3'] =term3
			df['term4'] =term4
			df['term5'] =term5

			fig = px.scatter(df, y="array_col", x="array_row", 
				color="region_ind", hover_data=['region', 'term1','term2', 'term3','term4', 'term5'], 
				width=400, height=400)
			fig.show()

	def plot_region_enrichment(self, top=3, flavor='default'):
		"""plot region enrichment barplot in either plotly flavor or default (seaborn) flavor
		
		Parameters:
		-----------
		top: int, number of top enriched terms to be plotted, default 3.
		flavor: str,
			- default: seaborn plot
			- plotly: plotly plot
		"""
		assert flavor in ['default', 'plotly']
		regions = self.region_enrichment_result.keys()
		region_dfs = []
		for region in regions:
			res = self.region_enrichment_result[region].sort_values(by=['Odds Ratio'], 
				ascending=False).iloc[:top,:]
			res['region'] = region
			region_dfs.append(res)
		region_dfs = pd.concat(region_dfs)
		region_dfs['-log10(P-adjust)'] =  - np.log10(10E-10 + region_dfs['Adjusted P-value'])
		region_dfs['log2(Odds-ratio)'] = np.log2(region_dfs['Odds Ratio']).astype('float')
		if flavor == 'plotly':
			fig = px.bar(region_dfs, x='-log10(P-adjust)', y="Term", 
				color="log2(Odds-ratio)", barmode="group",
				facet_col="region", height=800, width=1600)
			fig.show()
		else:
			f = plt.figure(figsize=(10, 5))
			sns.catplot(data=region_dfs, x='-log10(P-adjust)', y="Term", 
    		color="-log10(P-adjust)", palette='Reds', col='region',
            kind='bar')
			return f

	def plot_region_volcano(self, ncols=None, nrows=None):
		"""Plot volcano gene enrichment plot with top 10 genes listed.
		"""
		region_deg_results = self.region_deg_results
		regions = list(set(region_deg_results.region))

		sig_colors = {'NS': 'gray', 'UP': 'red', 'DOWN': 'blue'}

		if (ncols is None) or (nrows is None):
			nrows = 1
			ncols = len(regions)

		f, axs = plt.subplots(ncols=ncols, nrows=nrows, 
			figsize=(5 * ncols, 5 * nrows), 
			sharey=True, sharex=True)

		for k in range(len(regions)):
			region = regions[k]
			kdf = region_deg_results.loc[region_deg_results.region==region,:]
			kdf['-log10(FDR)'] = -1 * np.log10(kdf['padj'] + 1e-12)
			kdf['group'] = 'NS'
			kdf.loc[(kdf.padj <= 0.01) & (kdf.lfc > 0.26), 'group'] = 'UP'
			kdf.loc[(kdf.padj <= 0.01) & (kdf.lfc < -0.26), 'group'] = 'DOWN'
			sns.scatterplot(data=kdf, y='-log10(FDR)', x='lfc', hue='group', ax=axs[k], palette=sig_colors)
			axs[k].set_title(region)
			handles, labels = axs[k].get_legend_handles_labels()
			axs[k].legend().remove()
			axs[k].spines['right'].set_visible(False)
			axs[k].spines['top'].set_visible(False)
			
			kdf0 = kdf.loc[kdf.group == 'UP',:]
			kdf1 = kdf0.sort_values("-log10(FDR)", ascending=False).iloc[:10, :]
			ys, xs, genes = kdf1['-log10(FDR)'].to_numpy(), kdf1['lfc'].to_numpy(), kdf1['gene'].tolist()
			for i in range(len(genes)):
				axs[k].text(4, 12-(0.8*i), genes[i], fontsize=13, color='black')
			axs[k].set_xlabel("log2(fold-change)", fontsize=14)
			axs[k].set_ylabel("-log10(FDR)", fontsize=14)

		plt.subplots_adjust(wspace=0.2)
		plt.close()
		return f


	def impute(self, method='spKNN', n_neighbors=4, ncores=1, nExperts=5, layer='CPM', thre=0):
		"""Important function to impute the data. For more information, please investigate
		imputers.py and MIST2.py. Due to time constraints, default imputation apporach is spKNN.

		Parameters:
		-----------
		method: str, options included ["MAGIC", "knnSmooth", "mcImpute", "spKNN", "MIST"]
		n_neighbors: int, parameter for spKNN
		ncores: int, for parallel computing
		nExperts: int, for MIST
		thre: correlation threshold for determining neighors

		Return:
		-------
		Imputed daat as a Pandas data frame.
		"""
		rdata = self.shallow_copy()
		imputer = Imputer(method, rdata)
		nodes = self.nodes
		imputed_data = imputer.fit_transform(n_neighbors=n_neighbors, 
			ncores=ncores, nExperts=nExperts, layer=layer, thre=thre)
		return imputed_data

	def plot_region_boundaries(self, region_colors = None, by='UMI'):
		radius = 2
		alpha = 1 / (radius-0.1)

		f , ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
		region_df = self.adata.obs
		regions = set(region_df.region_ind)
		if 'isolated' in regions:
			regions.remove('isolated')
		regions = list(regions)
		
		xs, ys = region_df.array_col.tolist(), region_df.array_row.tolist()

		if by == 'UMI':
			sns.scatterplot(data=region_df, x='array_col', 
					y='array_row', hue='total_counts', palette='Reds', ax = ax)
		else:
			ax.scatter(xs, ys , color="lightgray", s=50)

		points = [[x,y] for x, y in zip(xs, ys)]

		if region_colors is None:
			region_colors = self.region_color_dict

		for region in regions:
			reg_df = region_df.loc[region_df.region_ind == region,:]
			xs, ys = reg_df.array_col.tolist(), reg_df.array_row.tolist()
			points = [[x,y] for x, y in zip(xs, ys)]
			
			if region_colors != None:
				c = region_colors[region]

			alpha_shape = alphashape(points, alpha)
			if by == 'UMI':
				ax.add_patch(PolygonPatch(alpha_shape, fc='none',
									  ec=c, linewidth=3))
			else:
				ax.scatter(xs, ys , color=c, alpha=0.2, s=50)
				ax.add_patch(PolygonPatch(alpha_shape, fc=c, alpha=0.5,
									  ec=c, linewidth=3))
		ax.axis('off')
		plt.gca().invert_yaxis()
		plt.close()
		return f

	def visualize_gene_expr(self, gene, region_colors = None,
				 gcmap='Reds', vmin=None, vmax = None):
		"""Visualize gene expression patterns as a heatmap with region boundaries in-situ.

		Parameters:
		-----------
		gene: str, gene name
		reigon_colors: dict, region color dictionary (optional).
		gcmap: str, cmap for gene
		vmin: float, value minimum for heatmap
		vmax: float, value maximum for heatmap

		Return:
		-------
		matplotlib figure object
		"""

		radius = 2
		alpha = 1 / (radius-0.1)
		f , ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
		region_df = self.adata.obs
		regions = set(region_df.region_ind)

		if "isolated" in regions:
			regions.remove('isolated')

		regions = list(regions)
		
		xs, ys = region_df.array_col.tolist(), region_df.array_row.tolist()
		gexpr = self.adata[:, gene].layers['CPM'].toarray()

		s = 50
		if self.adata.shape[0] > 1000:
			s = 30

		if vmin is not None and vmax is not None:
			sc = ax.scatter(xs, ys, c=gexpr, s=s, cmap=gcmap, vmin=vmin, vmax=vmax)
		else:
			sc = ax.scatter(xs, ys, c=gexpr, s=s, cmap=gcmap)

		plt.colorbar(sc)

		if region_colors is None:
			region_colors = self.region_color_dict

		points = [[x,y] for x, y in zip(xs, ys)]
		
		for region in regions:
			reg_df = region_df.loc[region_df.region_ind == region,:]
			xs, ys = reg_df.array_col.tolist(), reg_df.array_row.tolist()
			points = [[x,y] for x, y in zip(xs, ys)]
			
			c = region_colors[region]

			alpha_shape = alphashape(points, alpha)
			ax.add_patch(PolygonPatch(alpha_shape, fc='none', 
									  ec=c, linewidth=3))
		ax.axis('off')
		plt.gca().invert_yaxis()
		plt.close()
		return f
	
	def save(self, out_folder):
		"""Save the ReST object to designated folder

		Parameters:
		-----------
		out_folder: str, folder path to save the data to
		"""
		if not os.path.exists(out_folder):
			os.mkdir(out_folder)
		joblib.dump(self.adata.copy(), os.path.join(out_folder, "adata.job"))
		self.region_deg_results.to_csv(os.path.join(out_folder, "region_deg_results.csv"))
		self.thr_opt_fig.savefig(os.path.join(out_folder, "threshold_finding_figure.png"), dpi=200, bbox_inches='tight')

		atts = {'species': self.species,
			'shape': self.shape,
			'region_vAllmarker_dict': self.region_vAll_marker_dict,
			'region_color_dict':  self.region_color_dict,
			'auto_region_names': self.auto_region_names,
			'region_enrichment_result': self.region_enrichment_result
			}

		joblib.dump(atts, os.path.join(out_folder, 'ReST_attributes.job'))
	
	def load(self, folder):
		"""Load the ReST object from designated folder

		Parameters:
		-----------
		folder: str, folder path to load the data from
		"""
		assert os.path.exists(folder)
		self.adata = joblib.load(os.path.join(folder, "adata.job"))

		if os.path.exists(os.path.join(folder, "region_deg_results.csv")):
			self.region_deg_results = pd.read_csv(os.path.join(folder, "region_deg_results.csv"), index_col=0)

		atts = joblib.load(os.path.join(folder, 'ReST_attributes.job'))

		self.species = atts['species']
		self.shape = atts['shape']
		self.region_vAll_marker_dict = atts['region_vAllmarker_dict']
		self.auto_region_names = atts['auto_region_names']
		self.region_enrichment_result = atts['region_enrichment_result']
		self.region_color_dict = atts['region_color_dict']
	
	def manual_assign_region_names(self, region_names):
		""" Manually assign a name to each detected region. 

		Parameters:
		-----------
		region_names: a dictionary object containing region_ind to 
						manually matched names
		"""
		regions = set(self.adata.obs.region_ind)
		annot_regions = regions.intersection(set(region_names.keys()))
		assert len(annot_regions) > 0
		adata = self.adata.copy()
		adata = adata[adata.obs.region_ind.isin(annot_regions),:]
		self.manual_region_name_dict = region_names
		adata.obs['manual_name'] = adata.obs['region_ind'].map(region_names)
		self.annot_adata = adata

	def save_ReSort(self, folder, fmt='csv'):
		"""Save the detected regions as references,
			we recommend manually assign region names using 
			manual_assign_region_names() first.
		"""
		assert fmt in ['csv', 'tsv']
		sep = "," if fmt=='csv' else "\t"

		if self.annot_adata is None:
			adata = self.adata.copy()
			col = 'region_ind'
		else:
			adata = self.annot_adata.copy()
			col = 'manual_name'

		adata_ref = adata[adata.obs[col] != 'isolated', :]
		ref_meta = adata_ref.obs[[col]]
		ref_meta.columns = ['bio_celltype']

		ref_vals = adata_ref.raw.to_adata().X

		if not isinstance(ref_vals, np.ndarray):
			ref_vals = ref_vals.toarray()

		ref_vals = ref_vals.astype(int)
		ref_count = pd.DataFrame(data=ref_vals, index=adata_ref.obs_names, columns=adata_ref.var_names)
		ref_meta.to_csv(f"{folder}/ReSort_reference_meta.{fmt}", sep=sep)
		ref_count.to_csv(f"{folder}/ReSort_reference_raw_count.{fmt}", sep=sep)

		mixture_CPM =  self.adata.layers['CPM'].copy()
		if not isinstance(mixture_CPM, np.ndarray):
			mixture_CPM = mixture_CPM.toarray()
		mixture_CPM = mixture_CPM.astype(int)
		mixture_CPM = pd.DataFrame(data=mixture_CPM, index=self.adata.obs.new_idx, columns=self.adata.var_names)

		mixture_raw = self.adata.raw.to_adata().X
		if not isinstance(mixture_raw, np.ndarray):
			mixture_raw = mixture_raw.toarray()
		mixture_raw = pd.DataFrame(data=mixture_raw, index=self.adata.obs.new_idx, columns=self.adata.var_names)
		mixture_raw.to_csv(f"{folder}/ReSort_mixture_raw.{fmt}", sep=sep)
