
from utils import read_ST_data, data_norm
import neighbors

class Data(object):
	"""docstring for Data"""
	def __init__(self, countpath, radius=2, merge=5, norm="none"):
		self.filename = countpath
		self.norm = norm
		self.radius = radius
		count, meta = read_ST_data(self.filename)
		count.fillna(0, inplace=True)
		if norm != "none":
			count_matrix = data_norm(count_matrix, method=norm)

		self.count = count
		self.meta = meta
		self.nodes = neighbors.construct_graph(self.meta, self.radius)
		self.merge=5
		self.epsilon=0.6

	def update_ep(self, ep):
		self.epsilon=ep

		