import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, roc_auc_score, recall_score, precision_score
import sys
sys.path.append("../src/")
import utils
from utils import gene_density
from  tqdm import trange
from time import time
## Evaluate spot level performance for holdout test at log2 scale
def evalSpot(ori, mask, meta, model_data, model_name):
	spots = mask.index[(mask == 1).any(axis=0)]
	meta = meta.loc[spots,:]
	rmses, pccs_all, snrs, mapes = [], [], [], []
	spots_ho = []

	for i in trange(len(spots)):
		spot = spots[i]
		genes = mask.columns[mask.loc[spot,:] == 1].tolist()
		if len(genes) == 0:
			continue
		spots_ho.append(spot)
		tru = ori.loc[spot, genes].to_numpy()
		imp = model_data.loc[spot, genes].to_numpy()
		rmses.append(np.sqrt(np.mean(np.square(imp - tru))))
		pccs_all.append(pearsonr(imp, tru)[0])
		snrs.append(np.log2((np.sum(imp) +1) /
			(1+np.sum(np.absolute(tru-imp)))))
		mapes.append(np.mean(np.divide(np.absolute(imp - tru), tru)))
	meta = meta.loc[spots_ho,:]
	spot_perf = pd.DataFrame({"spot":spots_ho, "x":meta.iloc[:,0],
								"y":meta.iloc[:,1],
								"rmse":rmses,
								"pcc": pccs_all,
								"snr": snrs,
								"mape":mapes,
								"model":model_name})
	return spot_perf

## Evaluate gene level performance for holdout test
def evalGene(ori, mask, ho, meta, model_data, model_name):
	genes = mask.columns[(mask == 1).any(axis=1)]

	rmses, pccs_all, snrs, mapes = [], [], [], []
	mrs = []
	genes_ho = []

	for i in trange(len(genes)):
		gene = genes[i]
		spots = mask.index[mask.loc[:, gene] == 1].tolist()
		if len(spots) == 0:
			continue
		genes_ho.append(gene)
		mr = (ho.loc[:, gene] == 0).sum()/float(ho.shape[0])
		mrs.append(mr)
		tru = ori.loc[spots, gene].to_numpy()
		imp = model_data.loc[spots, gene].to_numpy()
		rmses.append(np.sqrt(np.mean(np.square(imp-tru))))
		pccs_all.append(pearsonr(imp,tru)[0])
		snrs.append(np.log2((np.sum(imp) +1) /
			(1+np.sum(np.absolute(tru-imp)))))
		mapes.append(np.mean(np.divide(np.absolute(imp - tru), tru)))

	gene_perf = pd.DataFrame({"gene": genes_ho,
				"rmse":rmses,
				"pcc": pccs_all,
				"snr": snrs,
				"mape":mapes,
				"model":model_name,
				"mr": mrs})
	return gene_perf

## Evaluate slide level performance for holdout test
def evalSlide(ori, mask, ho, model_data, model_name):
	M = np.ravel(mask)
	inds = np.where(M)
	tru = np.ravel(ori.values)[inds]
	imp = np.ravel(model_data.values)[inds]
	snr = np.log2(np.sum(imp) / np.sum(np.absolute(tru-imp)))
	rmse = np.sqrt(np.mean(np.square(imp - tru)))
	mape = np.mean(np.divide(np.absolute(imp - tru), tru))
	pcc = pearsonr(tru, imp)[0]
	MR1 = float((ho == 0).sum().sum()) / np.prod(ho.shape)
	MR2 = float((model_data == 0).sum().sum()) / np.prod(model_data.shape)
	perf_df = pd.DataFrame(data=[[rmse, mape, snr, pcc, model_name, MR1, MR2, MR1-MR2]],
			 columns= ['RMSE', 'MAPE', 'SNR', 'PCC', 'ModelName', 'hoMR', 'impMR', 'redMR'])
	return perf_df

## Function to evaluate all models for one dataset
def evalAll(data_folder, model_names, cvFold=5):
	# Slide performance
	model_perf_dfs = []
	spot_perf_dfs = []
	gene_perf_dfs = []
	for seed in range(cvFold):
		st = time()
		mask = pd.read_csv("%s/ho_mask_%d.csv" %(data_folder, seed), index_col=0)
		genes = mask.columns.tolist()
		ho = pd.read_csv("%s/ho_data_raw_%d.csv" %(data_folder, seed), index_col=0)
		ho = ho.loc[mask.index, genes] 
		ori, meta = utils.read_ST_data("%s/raw.csv" %data_folder)
		t1 = time()
		print("[Fold %d] Ground truth data loading elapsed %.1f seconds." %(seed, t1 - st))
		ori = ori.loc[mask.index, genes]
#		ori = np.log2(ori + 1) #CPM to logCPM
		for model_name in model_names:
			t2 = time()
			fn = "%s/%s_raw_%d.csv" %(data_folder, model_name, seed)
			model_data = pd.read_csv(fn, index_col=0)
			model_data = model_data.loc[ori.index, genes]
#			model_data = np.log2(model_data + 1)

			t3 = time()
			print("[Fold %d, %s] Model data loading elapsed %.1f seconds." %(seed, model_name, t3-t2))
			model_perf_df = evalSlide(ori, mask, ho, model_data, model_name)
			t4 = time()
			print("[Fold %d, %s] Slide-level performance evaluation elapsed %.1f seconds." %(seed, model_name, t4-t3))
			try:
				spot_perf_df = evalSpot(ori, mask, meta, model_data, model_name)
			except:
				spot_perf_df = model_perf_df
			t5 = time()
			print("[Fold %d, %s] Spot-level  performance evaluation elapsed %.1f seconds." %(seed, model_name, t5-t4))
			gene_perf_df = evalGene(ori, mask, ho,  meta, model_data, model_name)
			t6 = time()
			print("[Fold %d, %s] Gene-level  performance evaluation elapsed %.1f seconds." %(seed, model_name, t6-t5))
			model_perf_df['cvFold'] = seed
			model_perf_dfs.append(model_perf_df)
			spot_perf_df['cvFold'] = seed
			spot_perf_dfs.append(spot_perf_df)
			gene_perf_df['cvFold'] = seed
			gene_perf_dfs.append(gene_perf_df)

	model_perf_dfs = pd.concat(model_perf_dfs)
	spot_perf_dfs = pd.concat(spot_perf_dfs)
	gene_perf_dfs = pd.concat(gene_perf_dfs)
	return model_perf_dfs, spot_perf_dfs, gene_perf_dfs

### Main
def main(data_folder):
	## create a performance folder to save results if not exists
	perf_folder = os.path.join(data_folder, "performance")
	if not os.path.exists(perf_folder):
		os.mkdir(perf_folder)
	#model_names = ["spImpute", "mcImpute","MAGIC", "spKNN", "knnSmooth"]
	model_names = ["spImpute","mcImpute"]
	## get performance
	slidePerf, spotPerf, genePerf = evalAll(data_folder, model_names, 2)
	## save performance
	slidePerf.to_csv(os.path.join(perf_folder, "slide_level_results.csv"))
	spotPerf.to_csv(os.path.join(perf_folder, "spot_level_results.csv"))
	genePerf.to_csv(os.path.join(perf_folder, "gene_level_results.csv"))

if __name__ == "__main__":
	dataDir = sys.argv[1]
	main(dataDir)



