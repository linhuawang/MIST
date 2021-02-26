## scripts to generate holdout test data
import pandas as pd
import numpy as np
import sys

sys.path.append("../src/")
import utils
from time import time
from os.path import exists, join
from os import mkdir

def generate_cv_masks(original_data):
	np.random.seed(2021)
	# make a dumb hold out mask data
	ho_mask = pd.DataFrame(columns = original_data.columns,
					index = original_data.index,
					data = np.zeros(original_data.shape))
	# make templates for masks and ho data for 5 fold cross validation
	ho_masks = [ho_mask.copy() for i in range(5)]
	ho_dsets = [original_data.copy() for i in range(5)]
	# for each gene, crosss validate
	for gene in original_data.columns.tolist():
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
	norm_fn = join(data_folder, "norm.csv")
	raw_fn = join(data_folder, "raw.csv")
	original_data, meta_data = utils.read_ST_data(raw_fn)
	original_data = utils.filterGene_sparsity(original_data,filt)
	original_data =  utils.data_norm(original_data,method="cpm")
	original_data.to_csv(data_folder + "/norm.csv")

	print("Data normed and filtered: %s spots, %s genes." %(original_data.shape[0], original_data.shape[1]))
	# generate 5 fold cross validation datasets
	ho_dsets, ho_masks = generate_cv_masks(original_data)
	for fd in range(5):
		ho_data, ho_mask = ho_dsets[fd], ho_masks[fd]
		ho_data.to_csv(data_folder + "/ho_data_%d.csv" %fd)
		ho_mask.to_csv(data_folder + "/ho_mask_%d.csv" %fd)
		print("[Fold %d] Hold out data generated." %fd)

if __name__ == "__main__":
	folder = sys.argv[1]
	if "slideseq" in folder:
		main(folder, 0.2)
	else:
		main(folder, 0.5)
