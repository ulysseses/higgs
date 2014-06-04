import json
from feature_select import CustomParameterGrid, default_eval_params, default_preproc_params


search_space_path = "../working/cv_search_space_demo2.json"

def preproc_optimize(server, i_reduced, eval_params=default_eval_params):
	''' Optimize just the pre-processing parameters.

		`eval_params` can be a Python dict or string. If it is a 
			string, then `eval_params` will be loaded from the
			best_results.json file located in said string.
	'''
	
	# open params file
	with open(search_space_path, 'rb') as f:
		search_space = json.load(f)
	if type(eval_params) == str:
		filename = eval_params
		with open(filename, 'rb') as f:
			eval_params = json.load(f)['best_eval_params']

	best_score = -1
	best_preproc_params = None
	best_eval_params = None
	try:
		for preproc_params in CustomParameterGrid(search_space['preproc']):
			# pre-process
			server.preprocess(preproc_params)
			# evaluate
			score = server.evaluate(eval_params)
			if score > best_score:
				best_score = score
				best_preproc_params = server.get_best_preproc_params()
				best_eval_params = server.get_best_eval_params()
	except Exception, e:
		print "\n\n"
		print e
		print "current best score:", best_score
		print "current best preproc_params:", best_preproc_params
		print "current best eval_params:", best_eval_params
	
	best_score = server.get_best_score()
	best_preproc_params = server.get_best_preproc_params()
	best_eval_params = server.get_best_eval_params()

	# save best results to json file
	with open("../working/best_results_%d.json" % i_reduced, 'wb') as f:
		output = {
				"best_score": best_score,
				"best_preproc_params": best_preproc_params,
				"best_eval_params": best_eval_params
			}
		json.dump(output, f, indent=4)
	return output


def trainer_optimize(server, i_reduced=-1, preproc_params=default_preproc_params):
	''' Optimize just the trainer parameters.

		`preproc_params` can be a Python dict or a string. If it is a
			string, then `preproc_params` will be loaded from the 
			best_results.json file located in said string.
	'''
	# open params file
	with open(search_space_path, 'rb') as f:
		search_space = json.load(f)
	if type(preproc_params) == str:
		filename = eval_params
		if i_reduced == -1:
			i_reduced = int(filename[-6])  # %d.json
		with open(filename, 'rb') as f:
			preproc_params = json.load(f)['best_preproc_params']

	# pre-process data only once
	server.preprocess(preproc_params)
	
	best_score = -1
	best_preproc_params = None
	best_eval_params = None
	try:
		for eval_params in CustomParameterGrid(search_space['eval']):
			# evaluate
			score = server.evaluate(eval_params)
			if score > best_score:
				best_score = score
				best_preproc_params = server.get_best_preproc_params()
				best_eval_params = server.get_best_eval_params()
	except Exception, e:
		print "\n\n"
		print e
		print "current best score:", best_score
		print "current best preproc_params:", best_preproc_params
		print "current best eval_params:", best_eval_params
	
	best_score = server.get_best_score()
	best_preproc_params = server.get_best_preproc_params()
	best_eval_params = server.get_best_eval_params()

	# save best results to json file
	with open("../working/best_results_%d.json" % i_reduced, 'wb') as f:
		output = {
				"best_score": best_score,
				"best_preproc_params": best_preproc_params,
				"best_eval_params": best_eval_params
			}
		json.dump(output, f, indent=4)
	return output


if __name__ == '__main__':
	import sys
	import xmlrpclib
	host, port = "localhost", sys.argv[1]
	server = xmlrpclib.ServerProxy("http://%s:%d" % (host, port))
	i_reduced = sys.argv[2]
	print(server.system.listMethods())
	print(preproc_optimize(server, i_reduced))
	print(trainer_optimize(server, preproc_params="../working/best_results_%d.json" % \
		i_reduced))


