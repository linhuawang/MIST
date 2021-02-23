import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, roc_auc_score, recall_score, precision_score
import sys
import utils
from utils import gene_density

def spotPerformance(ori, mask, meta, model_data, model_name):
	spots = ori.index.tolist()
	meta = meta.loc[spots,:]
	rmses = []
	for spot in spots:
		genes = mask.columns[mask.loc[spot,:] == 1].tolist()
		tru = ori.loc[spot, genes].to_numpy()
		imp = model_data.loc[spot, genes].to_numpy()
		rmse = np.sqrt(np.mean(np.square(imp - tru)))
		rmses.append(rmse)
	spot_perf = pd.DataFrame({"x":meta.iloc[:,0],
								"y":meta.iloc[:,1],
								"rmse":rmses,
								"model":model_name})
	return spot_perf

def wholeSlidePerformance(ori, mask, ho, model_data, model_name):
	M = np.ravel(mask)
	inds = np.where(M)
	tru = np.ravel(ori.values)[inds]
	imp = np.ravel(model_data.values)[inds]
	snr = np.log2(np.sum(imp) / np.sum(np.absolute(tru-imp)))
	rmse = np.sqrt(np.mean(np.square(imp - tru)))
	mape = np.mean(np.divide(np.absolute(imp - tru), tru))
	pcc = pearsonr(tru, imp)
	MR1 = float((ho == 0).sum().sum()) / np.prod(ho.shape)
	MR2 = float((model_data == 0).sum().sum()) / np.prod(model_data.shape)
	perf_df = pd.DataFrame(data=[[rmse, mape, snr, pcc, model_name, MR1, MR2, MR1-MR2]],
						  columns= ['RMSE', 'MAPE', 'SNR', 'PCC', 'ModelName', 'hoMR', 'impMR', 'redMR'])
	return perf_df

def main(data_folder):
	model_names = []
	for epi in np.arange(-1., 1.04, 0.05):
		if epi >= 0:
			model = 'CCRMp%s' %str(int(epi*100))
		else:
			model = 'CCRMn%s' %str(int(epi*100*(-1)))
		model_names.append(model)
	model_names.append("mcPy")
	slidePerf, spotPerf = evaluateAll(data_folder, model_names)
	slidePerf.to_csv(data_folder + "/slidePerformance.csv")
	spotPerf.to_csv(data_folder + "/spotPerformance.csv")


def evaluateAll(data_folder, model_names):
	# Slide performance
	model_perf_dfs = []
	spot_perf_dfs = []
	for seed in range(1):
		mask = pd.read_csv("%s/ho_mask_%d.csv" %(data_folder, seed), index_col=0)
		genes = mask.columns
		mask = mask.loc[:, genes]
		ho = pd.read_csv("%s/ho_data_%d.csv" %(data_folder, seed), index_col=0)
		ho = ho.loc[:, genes]
		ori, meta = utils.read_ST_data("%s/CPM_filtered.csv" %data_folder)
		ori = ori.loc[:, genes]
		ori = np.log2(ori + 1)
		for model_name in model_names:
			if 'CCRM' not in model_name:
				fn = "%s/%s_%d.csv" %(data_folder, model_name, seed)
			else:
				fn = "%s/%s_%d" %(data_folder, model_name, seed)
			model_data = pd.read_csv(fn, index_col=0)
			model_data.columns = ori.columns
			model_data = np.log2(model_data + 1)
			model_perf_df = wholeSlidePerformance(ori, mask, ho, model_data, model_name)
			spot_perf_df = spotPerformance(ori, mask, meta, model_data, model_name)
			model_perf_df['cvFold'] = seed
			model_perf_dfs.append(model_perf_df)
			spot_perf_df['cvFold'] = seed
			spot_perf_dfs.append(spot_perf_df)

	model_perf_dfs = pd.concat(model_perf_dfs)
	spot_perf_dfs = pd.concat(spot_perf_dfs)

	return model_perf_dfs, spot_perf_dfs

if __name__ == "__main__":
	dataDir = sys.argv[1]
	main(dataDir)



