## scripts to generate holdout test data
import pandas as pd
import numpy as np
import sys
sys.path.append("../src/")
import utils
from time import time
from os.path import exists, join
from os import mkdir

def main(data_folder, filt=0.5, norm="cpm", kFold=5):
	norm_fn = join(data_folder, "norm.csv")
	raw_fn = join(data_folder, "raw.csv")
	raw, meta = utils.read_ST_data(raw_fn)

	#original_data = utils.filterGene_sparsity(original_data,filt)
	normed, libsize =  utils.data_norm(raw, norm)

	if norm in ["median", "logMed", "logCPM"]:
		normed = normed.round(2)

	normed.to_csv(join(data_folder,"norm.csv"))
	libsize.to_csv(join(data_folder, "libsize.csv"))

	normFilt = utils.filterGene_sparsity(normed,filt)
	genes = normFilt.columns.tolist()
	print("Holdouut %s genes." %(normFilt.shape[1]))
	# generate 5 fold cross validation datasets
	ho_dsets, ho_masks = utils.generate_cv_masks(normFilt, genes,5)

	for fd in range(kFold):
		ho_data, ho_mask = ho_dsets[fd], ho_masks[fd]
		ho_raw = utils.data_denorm(ho_data, join(data_folder, "libsize.csv"), norm)
		ho_data.to_csv(data_folder + "/ho_data_%d.csv" %fd)
		ho_raw.to_csv(join(data_folder, "ho_raw_%d.csv" %fd))
		ho_mask.to_csv(data_folder + "/ho_mask_%d.csv" %fd)
		print("[Fold %d] Hold out data generated, %d genes, %d spots." \
			%(fd, ho_data.shape[1], ho_data.shape[0]))

if __name__ == "__main__":
	folder = sys.argv[1]
	norm = sys.argv[2]
	assert norm in ["cpm", "logCPM", "median", "logMed"]
	if "slideseq" in folder:
		main(folder, 0.1,norm=norm)
	else:
		main(folder, 0.5,norm=norm)
