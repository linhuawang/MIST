import Data
import neighbors
import utils

data = Data.Data(countpath="/Users/linhuaw/Documents/spImpute/raw/Mel2_raw.csv", 
        epsilon=0.6, merge=0, norm="cpm")
ccs = neighbors.spatialCCs(data.nodes, data.cormat, 0.6, merge=0)
utils.get_true_zeros(data.count, data.meta, ccs)

