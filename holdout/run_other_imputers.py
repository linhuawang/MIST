from imputers import *
import sys
sys.path.append("../src/")
import utils
import Data
input_fn = sys.argv[1]
data = Data.Data(countpath = input_fn, norm="cpm")

for imputer_name in ["MAGIC", "knnSmooth", "mcImpute", "spKNN"]:
	out_fn = "_".join((input_fn.split("_")[:-1]  + [imputer_name + ".csv"]))
	print(out_fn)
	imputer = Imputer(imputer_name, data)
	iData = imputer.fit_transform()
	iData.to_csv(out_fn)