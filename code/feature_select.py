# FeatureSelector must be compatible with sklearn.pipeline.PipeLine
from interact import interact
from cfs import cfs
from itertools import product
from sklearn.ensemble import GradientBoostingClassifier


default_preproc_params = \
	{
		"discrete":				False,
		"spider":
			{
				"weak": 	True,
				"relabel": 	True,
				"p": 		1
			},
		"feature_selection":
			{
				"algorithm": "interact_cfs",
				"params":
					{
						"mode":			"uc",
						"threshold":	0.0001,
						"backward":		False,
						"look_ahead":	1
					}
			}
	}


default_eval_params = \
	{
		"xgb_params":
			{
				"silent":				1,
				"nthread":				10,
				"bst:eta":				0.1,
				"bst:max_depth":		6,
				"bst:min_child_weight":	2,
				"bst:subsample":		0.5,
				"objective":			"binary:logitraw",
				"num_round":			20,
				"eval_metric":			"ams@0.15"
			},
		"pos_balance_factor":		-1,
		"min_child_weight_ratio":	1,
		"cutoff_thresholds":		[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
	}


def CustomParameterGrid(search_space):
	primaries = []
	names = search_space.keys()
	for obj in search_space.values():
		if type(obj) != list:
			primaries.append([obj])
		else:
			primaries.append(obj)
	for param_combo in product(*primaries):
		out = {}
		gens = []
		gen_names = []
		for name, param_val in zip(names, param_combo):
			if type(param_val) == dict:
				gens.append(CustomParameterGrid(param_val))
				gen_names.append(name)
			else:
				out[name] = param_val
		if len(gens) == 0:
			yield out
		else:
			for param_combo in product(*gens):
				for name, param_val in zip(gen_names, param_combo):
					out[name] = param_val
				yield out


class FeatureSelector:
	def __init__(self, algorithm='cfs'):
		if algorithm == 'interact':
			self.fit = self.fit_interact
		elif algorithm == 'cfs':
			self.fit = self.fit_cfs
		elif algorithm == 'gbt':
			self.fit = self.fit_gbt
		elif algorithm == 'interact_cfs':
			self.fit = self.fit_interact_cfs
	
	def fit_interact(self, X, y, threshold=0):
		self.cols_ = interact(X, y, threshold)
	
	def fit_cfs(self, X, y, mode='uc', backward=True, look_ahead=1):
		self.cols_ = cfs(X, y, backward, look_ahead, mode)

	def fit_interact_cfs(self, X, y, mode='uc', threshold=0,
			backward=True, look_ahead=1, **kwargs):
		cols1 = interact(X, y, threshold)
		cols2 = cfs(X, y, backward, look_ahead, mode)
		cols_set = set(cols1) | set(cols2)
		cols = list(cols_set)
		cols.sort()
		self.cols_ = cols
	
	def fit_gbt(self, X, y, threshold=0, **kwargs):
		clf = GradientBoostingClassifier(**kwargs).fit(X, y)
		feature_importances = clf.feature_importances_
		feature_importances /= np.max(feature_importances)
		cols = [col for col, importance
			in enumerate(feature_importances)
			if importance >= threshold]
		self.cols_ = cols
	
	def transform(self, X):
		return X[:, self.cols_]
	
	def fit_transform(self, X, y, **kwargs):
		self.fit(X, y, **kwargs)
		return X[:, self.cols_]



