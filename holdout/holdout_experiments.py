from imputers import *
import os
import sys

if __name__ == '__main__':
	folder = sys.argv[1]
	ep = 0.6
	merge = 5
	radius = 2

	if "slideseq" in folder:
		merge = 50
		radius = 100

	for i in range(5):
		fn = os.path.join(folder, "ho_data_%d.csv" %i)
		data = Data(fn, radius=radius, merge=merge)
		for imputer_name in ["MAGIC", "knnSmooth", "mcImpute", "spKNN", "spImpute"]:
			out_fn = os.path.join(folder, "%s_%d.csv" %(imputer_name, i))
			imputer = Imputer(imputer_name, data)
			imputed_data = imputer.fit_transform()
			imputed_data.to_csv(os.path.join(folder, out_fn))
			print("[%d, %s] imputed %s." %(i, imputer_name, fn))

