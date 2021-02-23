## scripts to generate holdout test data
import pandas as pd
import numpy as np
import sys

sys.path.append("../src")
import utils
from time import time
from os.path import exists
from os import mkdir

def generate_cv_masks(original_data, genes):
	np.random.seed(2021)
	# make a dumb hold out mask data
	ho_mask = pd.DataFrame(columns = original_data.columns,
								index = original_data.index,
								data = np.zeros(original_data.shape))
	# make templates for masks and ho data for 5 fold cross validation
	ho_masks = [ho_mask.copy() for i in range(5)]
	ho_dsets = [original_data.copy() for i in range(5)]

	# for each gene, crosss validate
	for gene in genes:
		nonzero_spots = original_data.index[original_data[gene] > 0].tolist()
		np.random.shuffle(nonzero_spots)
		nspots = len(nonzero_spots)
		foldLen = int(nspots/5)
		for i in range(5):
			if i != 4:
				spot_sampled = nonzero_spots[i*foldLen: (i+1)*foldLen]
			else:
				spot_sampled = nonzero_spots[i*foldLen: ]
			ho_masks[i].loc[spot_sampled, gene] = 1
			ho_dsets[i].loc[spot_sampled, gene] = 0
	return ho_dsets, ho_masks

def main(data_folder, filt=0.5, read_mask='no'):
	if read_mask == 'yes':
		original_data, meta_data = utils.read_ST_data(data_folder + "/cpm.csv")
		ho_mask = pd.read_csv(data_folder + "/ho_data_%d.csv" %seed, index_col=0)
		ho_data = pd.read_csv(data_folder + "/ho_data_%d.csv" %seed, index_col=0)
	else:
		data_fn = data_folder + "/raw.csv"
		original_data, meta_data = utils.read_ST_data(data_fn)
		genes = utils.filterGene_sparsity(original_data,filt).columns.tolist()
		print(len(genes))
		if not exists(data_folder + "/cpm.csv"):
			original_data =  utils.cpm_norm(original_data)
			original_data.to_csv(data_folder + "/cpm.csv")
		# generate 5 fold cross validation datasets
		ho_dsets, ho_masks = generate_cv_masks(original_data, genes)
		for fd in range(5):
			ho_data, ho_mask = ho_dsets[fd], ho_masks[fd]
			ho_data.to_csv(data_folder + "/ho_data_%d.csv" %fd)
			ho_mask.to_csv(data_folder + "/ho_mask_%d.csv" %fd)


if __name__ == "__main__":
	folder = sys.argv[1]
	if "slideseq" in folder:
		main(folder, 0.2)
	else:
		main(folder, 0.5)
