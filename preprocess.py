import scanpy as sc
import anndata as ad
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import sys
from statsmodels.stats.multitest import multipletests
from scipy.stats import ranksums

print(sc.__version__, ad.__version__, pd.__version__, np.__version__, sns.__version__)

def preprocess(path=None, adata=None, counts=None, coordinates=None, 
	gene_df=None, species='Human', n_pcs=30, hvg_prop = 0.6):

	if path != None:
		adata = sc.read_visium(path)
	elif (counts != None) and (coordinates != None) and (gene_df != None):
		adata = ad.AnnData(X=counts, obs=coordinates, var=gene_df)
	elif adata != None:
		adata = adata
	else:
		print('Wrong data input. Either format of the following inputs are eligible:\n\
			\t 1. The out/ path from SpaceRanger results,\n\
			\t 2. Raw AnnData,\n\
			\t 3. A raw count matrix (numpy array), coordinates data frame, and gene information data frame.')
		sys.exit(0)

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
	#max_count = np.percentile(counts, 95)
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
	sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=int(0.6 * adata.shape[1]))
	## Construct a knn graph based on PC similarity (top 50, nneighbors=100)
	sc.pp.pca(adata, n_comps=30)
	adata.obsp['raw_weights'] = pd.DataFrame(data=adata.obsm['X_pca'], 
		index=adata.obs_names).T.corr().loc[adata.obs_names, adata.obs_names].values
	return adata




