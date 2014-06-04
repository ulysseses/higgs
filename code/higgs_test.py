#!/usr/bin/env python
'''
Load best parameters inferred from higgs_cv.py and create a Kaggle
prediction file.

Usage:
	python higgs_test.py [(reduced_data path) (cv_params path) (model_directory) (prediction output path)]
Examples:
	python higgs_test.py
	python higgs_test.py ../working/reduced_test.dat ../working/cv_params_demo.json ../working ../data/submission.csv
'''
import time
t0 = time.time()
import cPickle as pickle
import json
import os.path
import numpy as np
from interact import interact

# add path of xgboost python module
import inspect
import sys
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost/python")
sys.path.append(code_path)
import xgboost as xgb


# load reduced data from inpath
inpath = '../working/reduced_test.dat' if len(sys.argv) == 1 else sys.argv[1]
inpath = os.path.abspath(inpath)

# set model_directory
model_directory = '../working' if len(sys.argv) == 1 else sys.argv[3]
model_directory = os.path.abspath(model_directory)

# load best_record
best_records_path = "../working/best_records.json" if len(sys.argv) == 1 else sys.argv[2]
best_records_path = os.path.abspath(best_records_path)
with open(best_records_path, 'rb') as fp:
	best_records = json.load(fp)

# Load best cutoff_thresholds
with open(os.path.join(model_directory, "cols_cts.json"), 'rb') as fp:
	cols_cts = json.load(fp)

# Test
#test_sizes = [16713, 162712, 9665, 153003, 150463, 57444]
n_reduced_models = 6
res = []
for i_reduced in xrange(n_reduced_models):
	''' Load reduced data '''
	with open(inpath, 'rb') as f:
		for i in range(3*i_reduced):
			temp = pickle.load(f)
		Xd = pickle.load(f)
		Xc = pickle.load(f)
		events = pickle.load(f)
		temp = None  # prevent pickling memory leak (if it exists)
	
	''' Setup model and test '''
	cols, ct = cols_cts[i_reduced]
	record = best_records[i_reduced]
	discrete = record["discrete"]
	model_params = record["model_params"]
	nthread = model_params["nthread"]
	
	# discrete?
	if discrete:
		X = Xd
		del Xc
	else:
		X = Xc
	
	# INTERACT
	assert len(cols) != 0
	X = X[:, cols]
	
	xgmat = xgb.DMatrix(X)
	bst = xgb.Booster({'nthread': nthread})
	bst.load_model(os.path.join(model_directory, '%d.model' % i_reduced))
	y_pred = bst.predict(xgmat)
	for j in xrange(len(y_pred)):
		res.append((events[j], y_pred[j], ct))

# Write out predictions
outpath = "../data/submission.csv" if len(sys.argv) == 1 else sys.argv[4]
outfile = os.path.abspath(outpath)

rorder = {}
for k, v, tr in sorted( res, key = lambda x:-x[1] ):
    rorder[ k ] = len(rorder) + 1
fo = open(outfile, 'w')
nhit = 0
ntot = 0
fo.write('EventId,RankOrder,Class\n')
for k, v, tr in res:
	ntop = tr * len(rorder)
	if rorder[k] <= ntop:
		lb = 's'
		nhit += 1
	else:
		lb = 'b'        
	fo.write('%s,%d,%s\n' % ( k, len(rorder) + 1 - rorder[k], lb ) )
	ntot += 1
fo.close()
print "nhit:", nhit
print "ntot:", ntot




t1 = time.time()
print "Total Time elapsed: %d min %d sec" % ((t1-t0)//60, (t1-t0) - ((t1-t0)//60)*60)
