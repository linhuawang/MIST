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
from sklearn.preprocessing import StandardScaler
from alphashape import alphashape
from descartes import PolygonPatch

class ReST(object):
	"""docstring for ClassName"""
	def __init__(self, path=None, adata=None, 
		counts=None, coordinates=None, gene_df=None):
		# super(ClassName, self).__init__()
		if path != None:
			adata = sc.read_visium(path)
		elif (counts is not None) and (coordinates is not None) and (gene_df is not None):
			adata = ad.AnnData(X=csr_matrix(counts.values), obs=coordinates, var=gene_df)
		elif adata is not None:
			adata.X = csr_matrix(adata.X)
			adata = adata
		else:
			print('Wrong data input. Either format of the following inputs are eligible:\n\
				\t 1. The out/ path from SpaceRanger results,\n\
				\t 2. Raw AnnData,\n\
				\t 3. A raw count matrix (numpy array), coordinates data frame, and gene information data frame.')
			sys.exit(0)
		self.adata = adata
		self.nodes = None
		self.shape = self.adata.shape

	def shallow_copy(self):
		rd2 = ReST(adata=self.adata.copy())
		return rd2

	def preprocess(self, hvg_prop=0.8,species='Human', n_pcs=30, filter_spot=True):
		adata = self.adata.copy()
		adata.var_names_make_unique()
		adata.obs['new_idx'] = adata.obs[['array_col', 'array_row']].apply(lambda x: 'x'.join(x.astype(str)), axis=1)

		print(f'Before QC: {adata.shape[0]} observations and {adata.shape[1]} genes.')

		## Preprocessing and QC
		if species == 'Human':
			adata.var["mt"] = adata.var_names.str.startswith("MT-")
		elif species == 'Mouse':
			adata.var["mt"] = adata.var_names.str.startswith("mt-")
		else:
			print("Please pass a valid species name from ['Human', 'Mouse'].")

		sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
		counts = adata.obs.total_counts.to_numpy()
		min_count = np.percentile(counts, 2)
		if filter_spot:
			sc.pp.filter_cells(adata, min_counts=min_count)
		#sc.pp.filter_cells(adata, max_counts=max_count)
		adata = adata[adata.obs["pct_counts_mt"] < 20]
		# print(f"#cells after MT filter: {adata.n_obs}")
		sc.pp.filter_genes(adata, min_cells=10)
		print(f'After QC: {adata.shape[0]} observations and {adata.shape[1]} genes.')
		## Data normalization 
		adata.raw = adata
		# LOG2CPM normalization
		sc.pp.normalize_total(adata, inplace=True, target_sum=10**6)
		adata.layers['CPM'] = adata.X.copy()
		sc.pp.log1p(adata, base=2)
		sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=int(hvg_prop * adata.shape[1]))
		## Construct a knn graph based on PC similarity (top 50, nneighbors=100)
		sc.pp.pca(adata, n_comps=n_pcs)

		adata.obsp['raw_weights'] = pd.DataFrame(data=adata.obsm['X_pca'], 
			index=adata.obs_names).T.corr().loc[adata.obs_names, adata.obs_names].values
		self.adata=adata

	def extract_regions(self, min_sim=0.5, max_sim=0.91,
					 gap=0.05, min_region=40, sigma=1):

		warnings.filterwarnings('ignore')

		adata = self.adata.copy()
		mixture_meta = pd.DataFrame({'x':adata.obs.array_col.tolist(),
									 'y':adata.obs.array_row.tolist()}, 
										index=adata.obs['new_idx'])

		count_df = pd.DataFrame(data=adata.X.toarray(), 
						  index=adata.obs.new_idx.tolist(), 
						 columns=adata.var.index.tolist())

		cor_df = pd.DataFrame(data=adata.obsp['raw_weights'], 
						  index=adata.obs.new_idx.tolist(), 
						 columns=adata.obs.new_idx.tolist())
		t11 = time()
		count_data = Data(count=count_df, meta=mixture_meta, cormat=cor_df)
		t2 = time()
		print(f"MIST Data created in {(t2-t11):.2f} seconds.")
		results = select_epsilon(count_data, min_sim=min_sim, 
												max_sim=max_sim, gap=gap, 
												min_region=min_region, sigma=1)

		self.thr_opt_fig = results['thre_figure']
		sample_regions = results['region_df']
		count_data.epsilon = results['threshold']

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
		regions = set(self.adata.obs.region_ind)
		regions.remove("isolated")

		if region_colors is None:
			colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
						'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
						'dimgrey', 'silver', 'rosybrown', 'firebrick', 'mistyrose',
							'tan', 'yellowgreen', 'dodgerblue', 'magenta', 'deepskyblue']
			if len(regions) > 20:
				colors = ['black'] * len(regions)
				print("Too many regions detected, please increase threshold for parameter min_region.")
				sys.exit(0)

			colors = colors[:len(regions)]
			region_colors = dict(zip(regions, colors))
		region_colors['isolated'] = 'gray'
		self.region_color_dict = region_colors


	def extract_regional_markers(self, mode='all'):
		adata = self.adata.copy()
		regions = set(adata.obs.region_ind)
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
					#print(f"Region DEG {reg1}-{reg2}: {df.loc[(df.lfc > 0.26) & (df.padj <=0.01)].shape[0]}, Region DEG {reg2}-{reg1}: {df.loc[(df.lfc < -0.26) & (df.padj <=0.01)].shape[0]}")
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
		import gseapy as gp
		from gseapy.plot import barplot, dotplot
		print(f"Running GSEA on mode {mode} for species {species}.")

		if mode not in ['all', 'mutual']:
			print("Either mode 'all' or 'mutual' is accepted.")
			sys.exit(0)
		

		if gene_sets is not None:
			gene_sets = ['GO_Biological_Process_2021']

		if species not in ['Mouse', 'Human']:
			print(f"GSEA not supported for species {species}. Only 'Human' or 'Mouse' is accepted.")
			return

		region_inds = set(self.adata.obs.region_ind)
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

			res = gp.enrichr(gene_list= region_markers, 
							gene_sets='GO_Biological_Process_2021',
							organism=species)
			res = res.res2d
			res = res.sort_values("Combined Score")
			res = res.loc[res['Adjusted P-value'] <= 0.05]

			if not res.empty:
				assigned_name = f"{res.Term.tolist()[0]} ({region_markers[0]})"
			else:
				assigned_name = ', '.join(region_markers[0: np.min(5, len(region_markers))])

			region_names[region] = assigned_name
			region_gsea_dict[region] = res

		self.auto_region_names = region_names
		self.region_enrichment_result = region_gsea_dict
		if os.path.exists("Enrichr"):
			shutil.rmtree("Enrichr")

	def plot_regions(self, marker_size=50, mode='spot_level'):
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

			import plotly.express as px
			fig = px.scatter(df, y="array_col", x="array_row", 
				color="region_ind", hover_data=['region', 'term1','term2', 'term3','term4', 'term5'], 
				width=400, height=400)
			fig.show()

	def plot_region_enrichment(self, top=3, flavor='default'):
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
			import plotly.express as px
			fig = px.bar(region_dfs, x='-log10(P-adjust)', y="Term", 
				color="log2(Odds-ratio)", barmode="group",
				facet_col="region", height=800, width=1600)
			fig.show()
		else:
			fig = plt.figure(figsize=(20, 10))
			sns.catplot(data=region_dfs, x='-log10(P-adjust)', y="Term", 
    		color="log2(Odds-ratio)", palette='viridis', col='region',
            kind='bar',  col_wrap=3)
			plt.show()

	def plot_region_volcano(self, ncols=None, nrows=None):
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
				axs[k].text(4, 12-(0.8*i), genes[i], fontsize=13, color='orange')
			axs[k].set_xlabel("log2(fold-change)", fontsize=14)
			axs[k].set_ylabel("-log10(FDR)", fontsize=14)

		plt.subplots_adjust(wspace=0.2)
		plt.close()
		return f


	def impute(self, method='MIST', n_neighbors=4, ncores=1, nExperts=5):
		rdata = self.shallow_copy()
		imputer = Imputer(method, rdata)
		nodes = self.nodes
		imputed_data = imputer.fit_transform(n_neighbors=n_neighbors, 
			ncores=ncores, nExperts=nExperts, nodes=nodes)
		return imputed_data

	def plot_region_boundaries(self, region_colors = None, by='UMI'):
		radius = 2
		alpha = 1 / (radius-0.1)

		f , ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
		region_df = self.adata.obs
		regions = set(region_df.region_ind)
		regions.remove('isolated')
		regions = list(regions)
		
		xs, ys = region_df.array_row.tolist(), region_df.array_col.tolist()

		if by == 'UMI':
			sns.scatterplot(data=region_df, x='array_row', 
					y='array_col', hue='total_counts', palette='Reds', ax = ax)
		else:
			ax.scatter(xs, ys , color="lightgray", s=50)

		points = [[x,y] for x, y in zip(xs, ys)]

		if region_colors is None:
			region_colors = self.region_color_dict

		for region in regions:
			reg_df = region_df.loc[region_df.region_ind == region,:]
			xs, ys = reg_df.array_row.tolist(), reg_df.array_col.tolist()
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
		plt.close()
		return f

	def visualize_gene_expr(self, gene, region_colors = None, gcmap='Reds'):
		radius = 2
		alpha = 1 / (radius-0.1)
		f , ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
		region_df = self.adata.obs
		regions = set(region_df.region_ind)
		regions.remove('isolated')
		regions = list(regions)
		
		xs, ys = region_df.array_row.tolist(), region_df.array_col.tolist()
		gexpr = self.adata[:, gene].layers['CPM'].toarray()
		scaler = StandardScaler()
		gexpr = np.ravel(scaler.fit_transform(gexpr))

		s = 50
		if self.adata.shape[0] > 1000:
			s = 30
		sc = ax.scatter(xs, ys, c=gexpr, s=s, cmap=gcmap)
		plt.colorbar(sc)

		if region_colors is None:
			region_colors = self.region_color_dict

		points = [[x,y] for x, y in zip(xs, ys)]
		
		for region in regions:
			reg_df = region_df.loc[region_df.region_ind == region,:]
			xs, ys = reg_df.array_row.tolist(), reg_df.array_col.tolist()
			points = [[x,y] for x, y in zip(xs, ys)]
			
			c = region_colors[region]

			alpha_shape = alphashape(points, alpha)
			ax.add_patch(PolygonPatch(alpha_shape, fc='none', 
									  ec=c, linewidth=3))
		ax.axis('off')
		plt.close()
		return f

