import pandas as pd
from matplotlib import pyplot as plt
import sys

component_fn = sys.argv[1]
outfn = sys.argv[2]
component_df = pd.read_csv(component_fn, index_col=0)

f, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(4,4))

img1 = ax1.scatter(x=component_df.x.to_numpy(),
			y=component_df.y.to_numpy(),
			c=component_df.cluster.to_list(),
			s=30)

plt.setp(ax1, xticks=[], yticks=[])

plt.savefig(outfn, bbox_inches='tight')