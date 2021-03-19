import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, roc_auc_score, recall_score, precision_score
import sys
sys.path.append("../src/")
from spImpute import select_ep
import utils
from  tqdm import trange
from time import time
from os.path import join
import Data
from neighbors import spatialCCs
from multiprocessing import Pool
from glob import glob
## Evaluate spot level performance for holdout test at log2 scale
def evalSpot(ori, mask, meta, model_data, model_name):
	spots = mask.index[(mask == 1).any(axis=1)]
	meta = meta.loc[spots,:]
	rmses, pccs_all, snrs = [], [], []
	for i in trange(len(spots)):
		spot = spots[i]
		genes = mask.columns[mask.loc[spot,:] == 1].tolist()

		try:
			tru = ori.loc[spot, genes].to_numpy()
			imp = model_data.loc[spot, genes].to_numpy()
			rmses.append(np.sqrt(np.mean(np.square(imp - tru))))
			pccs_all.append(pearsonr(ori.loc[spot,:].to_numpy(),
					 model_data.loc[spot,:].to_numpy())[0])
			snrs.append(np.log2((np.sum(imp) +1) /
				(1+np.sum(np.absolute(tru-imp)))))
		except:
			rmses.append(np.nan)
			pccs_all.append(np.nan)
			snrs.append(np.nan)

	meta = meta.loc[spots,:]
	spot_perf = pd.DataFrame({"spot":spots, "x":meta.iloc[:,0],
				"y":meta.iloc[:,1],"rmse":rmses,
				"pcc": pccs_all,"snr": snrs,
				"model":model_name})
	return spot_perf
## Evaluate gene level performance for holdout test
def evalGene(ori, mask, ho, meta, model_data, model_name):
	genes = mask.columns[(mask == 1).any(axis=0)]

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

		pccs_all.append(pearsonr(ori.loc[:, gene].to_numpy(),
								model_data.loc[:, gene].to_numpy())[0])

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
def evalSlide(ori, mask, ho, model_data, model_name, spots=None):
	if spots != None:
		observed = ori.loc[spots,:]
		ho_mask = mask.loc[spots,:]
		ho_data = ho.loc[spots,:]
		imputed_data = model_data.loc[spots,:]
	else:
		observed = ori.copy()
		ho_data = ho.copy()
		imputed_data = model_data.copy()
		ho_mask = mask.copy()

	M = np.ravel(ho_mask)
	inds = np.where(M)
	tru = np.ravel(observed.values)[inds]
	imp = np.ravel(imputed_data.values)[inds]
	snr = np.log2(np.sum(imp) / np.sum(np.absolute(tru-imp)))
	rmse = np.sqrt(np.mean(np.square(imp - tru)))
	mape = np.mean(np.divide(np.absolute(imp - tru), tru))

	try:
		pcc = pearsonr(tru, imp)[0]
	except:
		pcc = None

	MR1 = float((ho_data == 0).sum().sum()) / np.prod(ho_data.shape)
	MR2 = float((imputed_data == 0).sum().sum()) / np.prod(imputed_data.shape)

	perf_df = pd.DataFrame(data=[[rmse, mape, snr, pcc, model_name, MR1, MR2, MR1-MR2]],
			 columns= ['RMSE', 'MAPE', 'SNR', 'PCC', 'ModelName', 'hoMR', 'impMR', 'redMR'])
	return perf_df

## Function to evaluate all models for one dataset
def evalAll(data_folder, model_names, slideonly=True, cvFold=5):
	# Slide performance
	model_perf_dfs = []
	spot_perf_dfs = []
	gene_perf_dfs = []
	for seed in range(cvFold):
		st = time()
		mask = pd.read_csv("%s/ho_mask_%d.csv" %(data_folder, seed), index_col=0)
		genes = mask.columns.tolist()
		# ho = pd.read_csv("%s/ho_data_%d.csv" %(data_folder, seed), index_col=0)
		# ho = ho.loc[mask.index, genes]
		#ori, meta = utils.read_ST_data("%s/raw.csv" %data_folder)
		ori, meta = utils.read_ST_data("%s/norm.csv" %data_folder)
		t1 = time()
		print("[Fold %d] Ground truth data loading elapsed %.1f seconds." %(seed, t1 - st))
		ori = ori.loc[mask.index, genes]
		ho = ori.copy()
		ho[mask==1] = 0
		#ori = np.log2(ori + 1) #CPM to logCPM
		for model_name in model_names:
			t2 = time()
			#fn = os.path.join(data_folder, "%s_raw_%d.csv" %(model_name, seed))
			fn = os.path.join(data_folder, "%s_%d.csv" %(model_name, seed))
			if model_name == "spImpute":
				fn = glob(os.path.join(data_folder, "%s_*_%d.csv" %(model_name, seed)))[0]
			model_data = pd.read_csv(fn, index_col=0)
			## SAVER imputed by R language, - became . in the header
			if model_name == "SAVER":
				model_data.columns = ho.columns.tolist()
				model_data.index = ho.index.tolist()

			model_data = model_data.loc[ori.index, genes]
			#model_data = np.log2(model_data + 1)
			t3 = time()
			print("[Fold %d, %s] Model data loading elapsed %.1f seconds." %(seed, model_name, t3-t2))
			model_perf_df = evalSlide(ori, mask, ho, model_data, model_name)
			t4 = time()
			print("[Fold %d, %s] Slide-level performance evaluation elapsed %.1f seconds." %(seed, model_name, t4-t3))
			model_perf_df['cvFold'] = seed
			model_perf_dfs.append(model_perf_df)
			
			if not slideonly:
				spot_perf_df = evalSpot(ori, mask, meta, model_data, model_name)
				t5 = time()
				print("[Fold %d, %s] Spot-level  performance evaluation elapsed %.1f seconds." %(seed, model_name, t5-t4))
				gene_perf_df = evalGene(ori, mask, ho,  meta, model_data, model_name)
				t6 = time()
				print("[Fold %d, %s] Gene-level  performance evaluation elapsed %.1f seconds." %(seed, model_name, t6-t5))
				spot_perf_df['cvFold'] = seed
				spot_perf_dfs.append(spot_perf_df)
				gene_perf_df['cvFold'] = seed
				gene_perf_dfs.append(gene_perf_df)

	model_perf_dfs = pd.concat(model_perf_dfs)
	print(model_perf_dfs)
	model_perf_dfs.to_csv(os.path.join(data_folder, "slide_level_results.csv"))
	if not slideonly:
		spot_perf_dfs = pd.concat(spot_perf_dfs)
		gene_perf_dfs = pd.concat(gene_perf_dfs)
		spot_perf_dfs.to_csv(os.path.join(data_folder, "spot_level_results.csv"))
		gene_perf_dfs.to_csv(os.path.join(data_folder, "gene_level_results.csv"))


def eval_LCN_runner(param):
	dn, fd = param
	models = ["spImpute", "mcImpute", "MAGIC", "knnSmooth","spKNN", "SAVER"]
	projDir = "/houston_20t/alexw/ST/data/holdout_test/cpm_filtered"
	LCN_spots = LCN_captured_spots(join(projDir, dn), fd)

	with open(join(join(projDir, dn), "LCN_spots_%d.csv" %fd), "w") as f:
		f.write(",".join(LCN_spots))

	ho = pd.read_csv(join(join(projDir, dn), "ho_data_%d.csv" %fd), index_col=0)
	mask = pd.read_csv(join(join(projDir, dn), "ho_mask_%d.csv" %fd), index_col=0)
	observed = pd.read_csv(join(join(projDir, dn), "norm.csv"), index_col=0)
	observed = np.log2(observed + 1)
	results = []
	for model in models:
		model_df = pd.read_csv(join(join(projDir, dn), "%s_%d.csv" %(model, fd)), index_col=0)
		model_df = np.log2(model_df + 1)
		model_perf = evalSlide(observed, mask, ho, model_df, model, spots=LCN_spots)
		results.append(model_perf)
	results = pd.concat(results)
	results["cvFold"] = fd
	results["data"] = dn
	print(results)
	return results

def eval_LCNs():
	projDir = "/houston_20t/alexw/ST/data/holdout_test/cpm_filtered"
	#projDir = "~/Documents/spImpute/paper_data/holdout_test/"
	data_names = ["MouseWT", "MouseAD", "Melanoma2", "Prostate"]
	folds = range(5)
	params = []
	for dn in data_names:
		for fd in folds:
			params.append([dn, fd])
	p = Pool(10)
	model_perfs = p.map(eval_LCN_runner, params)
	model_perfs = pd.concat(model_perfs)
	model_perfs.to_csv(join(projDir, "LCNspots_slide_level_results.csv"))


def LCN_captured_spots(folder, fd):
	cp = join(folder, "ho_data_%d.csv" %fd)
	ho_data_obj = Data.Data(countpath=cp,radius=1.9, merge=0)
	ep = select_ep(ho_data_obj.count, ho_data_obj.meta, ho_data_obj.cormat)
	ho_data_obj.update_ep(ep)
	ccs = spatialCCs(ho_data_obj.nodes,ho_data_obj.cormat,
		ho_data_obj.epsilon, merge=0)
	spots = []

	for cc in ccs:
		if len(cc) > 5:
			for node in cc:
				spots.append(node.name)
	return spots

### Main
def main(data_folder, cvFold):
	## create a performance folder to save results if not exists
	perf_folder = os.path.join(data_folder, "performance")
	if not os.path.exists(perf_folder):
		os.mkdir(perf_folder)
	model_names = ["spImpute", "mcImpute","MAGIC", "spKNN", "knnSmooth"]
	#model_names = ["spImpute","mcImpute"]
	## get performance
#	slidePerf = evalAll(data_folder, model_names, 5)
	evalAll(data_folder, model_names, cvFold=cvFold)
	## save performance


if __name__ == "__main__":
# 	proj_dir = "/houston_20t/alexw/ST/data/holdout_test/logMedian"
# 	dns = ["MouseWT", "MouseAD", "Melanoma", "Prostate"]
# #	dns = ['Melanoma']
# 	for dn in dns:
# 		dataDir = join(proj_dir, dn)
# 		main(dataDir)
# 		print(dataDir)
	dataDir = sys.argv[1]
	if len(sys.argv) > 2:
		cvFold = int(sys.argv[2])
	else:
		cvFold = 5
	main(dataDir, cvFold)



