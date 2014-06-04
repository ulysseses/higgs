#!/usr/bin/env python
'''
Load best parameters inferred from higgs_cv.py and create a Kaggle
prediction file.

Usage:
	python higgs_train.py [(reduced_data path) (cv_params path) (best_params path) (model_directory)]
Examples:
	python higgs_train.py
	python higgs_train.py ../working/reduced_test.dat ../working/cv_params_demo.json ../working/best_params.json ../working
'''
import time
t0 = time.time()
import cPickle as pickle
import json
import os.path
import numpy as np
from interact import interact
from classification_metric import search_best_score, fbeta, f1

# add path of xgboost python module
import inspect
import sys
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost/python")
sys.path.append(code_path)
import xgboost as xgb


# load reduced data from inpath
inpath = '../working/reduced_train.dat' if len(sys.argv) == 1 else sys.argv[1]
inpath = os.path.abspath(inpath)

# set model_directory
model_directory = '../working' if len(sys.argv) == 1 else sys.argv[4]
model_directory = os.path.abspath(model_directory)

# load best_record
best_records_path = "../working/best_records.json" if len(sys.argv) == 1 else sys.argv[1]
best_records_path = os.path.abspath(best_records_path)
with open(best_records_path, 'rb') as fp:
	best_records = json.load(fp)

# Train
test_sizes = [16713, 162712, 9665, 153003, 150463, 57444]
n_reduced_models = 2
best_cols, best_cts = [], []
for i_reduced in xrange(n_reduced_models):
	''' Load test_size and reduced data'''
	test_size = test_sizes[i_reduced]
	with open(inpath, 'rb') as f:
		for i in range(4*i_reduced):
			temp = pickle.load(f)
		Xd = pickle.load(f)  # discrete
		Xc = pickle.load(f)  # continuous
		y = pickle.load(f)
		w = pickle.load(f) * float(test_size) / len(y)
		temp = None  # prevent pickling mem leak
	
	''' Load model parameters and train'''
	record = best_records[i_reduced]
	discrete = record["discrete"]
	it = record["interact_threshold"]
	model_params = record["model_params"]
	n_trees = model_params["n_trees"]
	pos_weight_ratio = model_params["pos_weight_ratio"]
	subparams = {}
	subparams["eta"] = model_params["bst:eta"]
	subparams["max_depth"] = model_params["bst:max_depth"]
	subparams["eval_metric"] = model_params["eval_metric"]
	subparams["objective"] = model_params["objective"]
	subparams["nthread"] = model_params["nthread"]
	subparams["silent"] = model_params["silent"]
	
	# discrete?
	if discrete:
		X = Xd
		del Xc
	else:
		X = Xc
	
	# INTERACT
	cols = interact(Xd, y, it)
	cols.sort()
	assert len(cols) != 0
	X = X[:, cols]
	
	xgmat = xgb.DMatrix(X, label=y, weight=w)
	if pos_weight_ratio != 0:  # positive example re-weighting
		sum_wpos = np.sum(w[i] for i in xrange(len(y)) if y[i] == 1)
		sum_wneg = np.sum(w[i] for i in xrange(len(y)) if y[i] == 0)
		subparams["scale_pos_weight"] = pos_weight_ratio * sum_wneg/sum_wpos
	plst = subparams.items()
	watchlist = [(xgmat, 'train')]
	bst = xgb.train(plst, xgmat, n_trees, watchlist)
	
	''' Find the best cutoff_threshold '''
	xgmat = xgb.DMatrix(X)
	y_pred = bst.predict(xgmat)
	cutoff_thresholds = record["cutoff_thresholds"]
	best_score, best_ct = search_best_score(y, y_pred, w, cutoff_thresholds)
	#best_score, best_ct = search_best_score(y, y_pred, None, cutoff_thresholds, fbeta)
	print "%dth model score: %.2f" % (i_reduced, best_score)
		
	''' Save model '''
	bst.save_model(os.path.join(model_directory, "%d.model" % i_reduced))
	print "%d.model saved" % i_reduced
	
	''' Save cols and cutoff_thresholds '''
	best_cols.append(cols)
	best_cts.append(best_ct)

# finally, save the best cols & thresholds into a json file for reading later
with open(os.path.join(model_directory, "cols_cts.json"), 'wb') as fp:
	obj = [(cols, ct) for (cols, ct) in zip(best_cols, best_cts)]
	json.dump(obj, fp, indent=4)


t1 = time.time()
print "Total Time elapsed: %d min %d sec" % ((t1-t0)//60, (t1-t0) - ((t1-t0)//60)*60)
