## scripts to generate holdout test data
import pandas as pd
import numpy as np
import sys
sys.path.append("../src/")
import utils
from spImpute import generate_cv_masks
from time import time
from os.path import exists, join
from os import mkdir

def main(data_folder, filt=0.5, norm="logMedian", kFold=5):
	norm_fn = join(data_folder, "norm.csv")
	raw_fn = join(data_folder, "raw.csv")
	original_data, meta_data = utils.read_ST_data(raw_fn)
	print(original_data.shape)
	#original_data = utils.filterGene_sparsity(original_data,filt)
	original_data =  utils.data_norm(original_data, norm).round(2)
	original_data = utils.filterGene_sparsity(original_data,filt)
	original_data.to_csv(data_folder + "/norm.csv")
	print("Data normed and filtered: %s spots, %s genes." %(original_data.shape[0], original_data.shape[1]))
	# generate 5 fold cross validation datasets
	ho_dsets, ho_masks = generate_cv_masks(original_data, 5)

	for fd in range(kFold):
		ho_data, ho_mask = ho_dsets[fd], ho_masks[fd]
		ho_data.to_csv(data_folder + "/ho_data_%d.csv" %fd)
		ho_mask.to_csv(data_folder + "/ho_mask_%d.csv" %fd)
		print("[Fold %d] Hold out data generated." %fd)

if __name__ == "__main__":
	folder = sys.argv[1]
	if "slideseq" in folder:
		main(folder, 0.1,norm="logMed")
	else:
		main(folder, 0.5,norm="logMed")
