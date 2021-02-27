
#from utils import read_ST_data, data_norm, spot_PCA_sims
import neighbors
import utils
import pandas as pd

class Data(object):
	"""docstring for Data"""
	# either countpath or (count and meta) have to be provided
	def __init__(self, count=None, meta=None, cormat=None,
				 countpath="", radius=2, merge=5, norm="none", epsilon=0.6):
		self.norm = norm
		## add count and meta data
		if countpath != "":
			self.filename = countpath
			count, meta = utils.read_ST_data(self.filename)
			count.fillna(0, inplace=True)
		else:
			assert isinstance(count, pd.DataFrame) and isinstance(meta, pd.DataFrame)

		if self.norm != "none":
			count = utils.data_norm(count, method=self.norm)
		self.count = count
		self.meta = meta

		if isinstance(cormat, pd.DataFrame):
			self.cormat = cormat
		else:
			self.cormat = utils.spot_PCA_sims(self.count)

		# add other features
		self.radius = radius
		self.nodes = neighbors.construct_graph(self.meta, self.radius)
		self.merge= merge
		self.epsilon=epsilon

	def update_ep(self, ep):
		self.epsilon=ep

	def update_cormat(self, cormat):
		self.cormat = cormat
