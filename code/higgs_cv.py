#!/usr/bin/env python
'''
Script to perform grid search over model parameters evaluated with cross-validation.
Uses a combination of xgboost and sklearn

Usage:
	python higgs_cv.py
'''
import time
t0 = time.time()
import cPickle as pickle
import json
import os.path
from itertools import islice
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.grid_search import ParameterGrid
from interact import interact
from classification_metric import search_best_score, precision

# add path of xgboost python module
import inspect
import sys
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost/python")
sys.path.append(code_path)
import xgboost as xgb


# load reduced data from inpath
inpath = '../working/reduced_train.dat'
inpath = os.path.abspath(inpath)

# load param_grid from cv_params_path
cv_params_path = "../working/cv_params_demo.json"
cv_params_path = os.path.abspath(cv_params_path)
with open(cv_params_path, 'rb') as fp:
	params_lst = json.load(fp)

# Cross Validation
#test_sizes = [16713, 162712, 9665, 153003, 150463, 57444]
test_size = 550000
n_reduced_models = 2
reduced_scores = []
for i_reduced in xrange(n_reduced_models):
	''' Load test_size, reduced data, and setup CV '''
	#test_size = test_sizes[i_reduced]
	with open(inpath, 'rb') as f:
		for i in range(4*i_reduced):
			temp = pickle.load(f)
		Xd = pickle.load(f)  # discrete
		Xc = pickle.load(f)  # continuous
		y = pickle.load(f)
		w = pickle.load(f)
		temp = None  # prevent pickling mem leak
	cv = KFold(len(y), n_folds=5)
	
	''' Exhaustive Search '''
	params = params_lst[i_reduced]
	records = []
	for discrete in params["discrete"]:
		# discrete?
		if discrete:
			Xt = Xd
		else:
			Xt = Xc
		for t,it in enumerate(params["interact_threshold"]):
			# INTERACT
			cols = interact(Xd, y, it)
			cols.sort()
			if len(cols) == 0: continue
			X = Xt[:, cols]
			xgmats = []
			for train, valid in cv:
				Xcv = X[train]
				ycv = y[train]
				wcv = w[train] * float(test_size) / len(ycv)  # ASM normalization
				xgmats.append(xgb.DMatrix(Xcv, label=ycv, weight=wcv))
			
			for p,model_params in enumerate(ParameterGrid(params["param_grid"])):
				n_trees = model_params["n_trees"]
				pos_weight_ratio = model_params["pos_weight_ratio"]
				subparams = {}
				subparams["eta"] = model_params["bst:eta"]
				subparams["max_depth"] = model_params["bst:max_depth"]
				subparams["eval_metric"] = model_params["eval_metric"]
				subparams["objective"] = model_params["objective"]
				subparams["nthread"] = model_params["nthread"]
				subparams["silent"] = model_params["silent"]
				
				split_scores = []
				for k,(train, valid) in enumerate(cv):
					''' Train '''
					print
					xgmat = xgmats[k]
					if pos_weight_ratio != 0:  # positive example re-weighting
						sum_wpos = np.sum(wcv[i] for i in xrange(len(ycv)) if ycv[i] == 1)
						sum_wneg = np.sum(wcv[i] for i in xrange(len(ycv)) if ycv[i] == 0)
						subparams["scale_pos_weight"] = pos_weight_ratio * sum_wneg/sum_wpos
						print "wpos=%.2f, wneg=%.2f, ratio=%.2f" % \
							(sum_wpos, sum_wneg, subparams["scale_pos_weight"])
					print "i%d,t%d,p%d,k%d n_trees=%d, eta=%.2f, max_depth=%d" % \
						(i_reduced, t, p, k, n_trees, subparams["eta"], subparams["max_depth"])
					plst = subparams.items()
					watchlist = [(xgmat, 'train')]
					bst = xgb.train(plst, xgmat, n_trees, watchlist)
					
					''' Validate '''
					Xcv = X[valid]
					ycv = y[valid]
					wcv = w[valid] * float(test_size) / len(ycv)
					xgmat = xgb.DMatrix(Xcv)
					y_pred = bst.predict(xgmat)
					# search best cutoff_threshold and record score
					cutoff_thresholds = params["cutoff_thresholds"]
					split_score, split_ct = search_best_score(ycv, y_pred, wcv, cutoff_thresholds)
					#split_score, split_ct = search_best_score(ycv, y_pred, None, cutoff_thresholds, precision)
					split_scores.append(split_score)
				cv_score_mean = np.mean(split_scores)
				cv_score_std = np.std(split_scores, ddof=1)  # unbiased
				
				''' Record '''
				record = dict()
				record["i_reduced"] = i_reduced
				record["discrete"] = discrete
				record["interact_threshold"] = it
				record["model_params"] = model_params
				record["cutoff_thresholds"] = params["cutoff_thresholds"]
				record["cv_score_mean"] = cv_score_mean
				record["cv_score_std"] = cv_score_std
				records.append(record)
	reduced_scores.append(records)

# Find "optimum" parameters based on cv_score_mean - 0.25*cv_score_std
calc_score = lambda record: record["cv_score_mean"] - 0.25*record["cv_score_std"]
find_max_score = lambda records: max(records, key=calc_score)
best_records = [find_max_score(records) for records in reduced_scores]

# Save best_record
outpath = "../working/best_records.json"
outpath = os.path.abspath(outpath)
with open(outpath, 'wb') as fp:
	json.dump(best_records, fp, indent=4)
	print "best records written to %s" % outpath




t1 = time.time()
print "Total Time elapsed: %d min %d sec" % ((t1-t0)//60, (t1-t0) - ((t1-t0)//60)*60)
