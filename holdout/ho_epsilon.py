
from os.path import join

if __name__ == "__main__":
	projDir = sys.argv[1]
	data_names = ["MouseWT", "MouseAD", "Melanoma1", "Melanoma2", "Prostate"]
	folds = range(5)
	for dn in data_names:
		for fd in folds:
			LCN_captured_spots(join(projDir, dn), fd)
