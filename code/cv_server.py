import cPickle as pickle
import json
import numpy as np
from spider import spider
from feature_select import FeatureSelector
from classification_metric import search_best_score
from sklearn.cross_validation import KFold

# add path of xgboost python module
import inspect, sys, os
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost/python")
sys.path.append(code_path)
import xgboost as xgb


class CrossValidator:
	def __init__(self, i_reduced=0, outofcore=False):
		self.i_reduced = i_reduced
		self.outofcore = outofcore
		if ~outofcore:
			with open("../working/reduced_train.dat", 'rb') as f:
				# Load pickled data for i_reduced
				self.data = []
				for i in range(6*4):
					if i / 4 == i_reduced:
						self.data.append(pickle.load(f))
		
		# cache
		self.reset_params()
	
	def reset_params(self):
		self.preproc_params = None
		self.best_preproc_params = None
		self.eval_params = None
		self.best_eval_params = None
		self.best_score = -1
	
	def reload_cv(self):
		# if the following doesn't create physical duplicates, then it's a-okay
		self.xgmats_train = []
		self.xgmats_valid = []
		self.w_10s = []
		self.pos_ratios = []

		self.ys_valid = []
		self.ws_valid = []
		
		self.cv = KFold(len(self.y), n_folds=5)
		for train,valid in self.cv:
			# train
			X = self.X[train]
			y = self.y[train]
			w = self.w[train]
			w *= 550000.0 / len(w)

			self.xgmats_train.append(xgb.DMatrix(X, label=y, weight=w))

			self.w_10s.append(np.mean(w) - 2*np.sted(w, ddof=1))

			sum_wpos = np.sum(w[i] for i in range(len(y)) if y[i] == 1)
			sum_wneg = np.sum(w[i] for i in range(len(y)) if y[i] == 0)
			pos_ratio = float(sum_wneg / sum_wpos)
			self.pos_ratios.append(pos_ratio)

			# validation
			X = self.X[valid]
			y = self.y[valid]
			w = self.w[valid]
			w *= 550000.0 / len(w)

			self.xgmats_valid.append(xgb.DMatrix(X))
			self.ys_valid.append(y)
			self.ws_valid.append(w)
	
	def preprocess(self, preproc_params=None):
		''' Pre-train the data with the provided preproc_params '''
		if preproc_params == None:
			preproc_params = self.preproc_params
		elif self.preproc_params == None:
			self.preproc_params = preproc_params
			skip_flag = False
		else:
			skip_flag = True
		
		p = preproc_params  # alias
		# discrete is no-op only for in-core; otherwise, skip_flag if same params
		discrete = p['discrete']
		skip_flag &= i_reduced == self.preproc_params['i_reduced']
		skip_flag &= discrete == self.preproc_params['discrete']
		if ~self.outofcore:
			self.Xd = self.data[0]
			self.Xc = self.data[1]
			self.y = self.data[2]
			self.w = self.data[3]
		elif ~skip_flag:
			i_reduced = self.i_reduced
			with open("higgs/working/reduced_train.dat", 'rb') as f:
				for i in range(4*i_reduced):
					temp = pickle.load(f)
				self.Xd = pickle.load(f)
				self.Xc = pickle.load(f)
				self.y = pickle.load(f)
				self.w = pickle.load(f)
				temp = None  # prevent pickling memory leak
		if discrete:
			self.X = self.Xd
		else:
			self.X = self.Xc
		if ~skip_flag:
			self.reload_cv()
		
		# skip_flag spider if same params
		skip_flag &= p["spider"] == self.preproc_params["spider"]
		if ~skip_flag:
			spider_params = p["spider"]
			spider_params['metric'] = 'wminkowski'
			spider_params['w'] = np.max(self.X, axis=0)
			self.X, self.y, self.w = spider(self.X, self.y, self.w, **spider_params)
		
		# skip_flag feature_selection if same params
		skip_flag &= p["feature_selection"] == self.preproc_params["feature_selection"]
		if ~skip_flag:
			fs_alg = p["feature_selection"]["algorithm"]
			fs_params = p["feature_selection"]["params"]
			fs = FeatureSelector(algorithm=fs_alg)
			fs.fit(self.Xd, self.y, **fs_params)
			self.X = fs.transform(self.X)
		
		self.preproc_params = p

	def evaluate(self, eval_params=None):
		''' Return the pessimistic bias of the average CV score by
			evaluating with the provided eval_params '''
		if eval_params == None:
			eval_params = self.eval_params
		elif self.eval_params == None:
			self.eval_params = eval_params
		
		p = eval_params  # alias
		xgb_params = p['xgb_params']
		pos_balance_factor = p['pos_balance_factor']
		min_child_weight_ratio = p['min_child_weight_ratio']
		cutoff_thresholds = p['cutoff_thresholds']
		
		split_scores = []
		for k in range(5):
			# manual positive example re-weighting
			pos_ratio = self.pos_ratios[k]
			if pos_balance_factor != -1:
				xgb_params['scale_pos_weight'] = pos_balance_factor * pos_ratio
			else:
				try:
					del xgb_params['scale_pos_weight']
				except:
					print "herp"
					pass
			# manual minimum child weight setup
			w_10 = self.w_10s[k]
			if min_child_weight_ratio != -1:
				if pos_balance_factor != -1:
					xgb_params['min_child_weight'] = \
						w_10 * pos_balance_factor * min_child_weight_ratio
				else:
					xgb_params['min_child_weight'] = \
						w_10 * min_child_weight_ratio
			else:
				try:
					del xgb_params['min_child_weight']
				except:
					print "asdf"
					pass
			# train
			watchlist = [(self.xgmats_train[k], 'train')]
			bst = xgb.train(xgb_params, self.xgmats_train[k],
				xgb_params['num_round'], watchlist)
			# validate
			y_rank = bst.predict(self.xgmats_valid[k])
			y_true = self.ys_valid[k]
			w = self.ws_valid[k]
			split_score, split_ct = search_best_score(y_true, y_rank,
				w, cutoff_thresholds)
			split_scores.append(split_score)
		score = float(np.mean(split_scores) - 0.25*np.std(split_scores, ddof=1))
		if score > self.best_score:
			self.best_score = score
			self.best_preproc_params = self.preproc_params
			self.best_eval_params = p
		
		self.eval_params = p
		return score

	def get_preproc_params(self):
		return self.preproc_params

	def get_eval_params(self):
		return self.eval_params
	
	def get_best_score(self):
		return self.best_score
		
	def get_best_preproc_params(self):
		return self.best_preproc_params
	
	def get_best_eval_params(self):
		return self.best_eval_params


if __name__ == '__main__':
	import sys
	import SimpleXMLRPCServer
	host, port = "localhost", sys.argv[1]
	server = SimpleXMLRPCServer.SimpleXMLRPCServer((host, port), allow_none=True)
	i_reduced = sys.argv[2]
	server.register_instance(CrossValidator(i_reduced=i_reduced, outofcore=False))
	server.register_multicall_functions()
	server.register_introspection_functions()
	print "XMLRPC Server is starting at %s, %s" % (host, port)
	server.serve_forever()

