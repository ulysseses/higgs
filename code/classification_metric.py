import numpy as np

def asm(ground_truth, predictions, weights):
	br = 10
	s = np.sum(weights[i] for i in xrange(len(ground_truth)) \
			   if (ground_truth[i] == 1) and (predictions[i] == 1))
	b = np.sum(weights[i] for i in xrange(len(ground_truth)) \
			   if (ground_truth[i] == 0) and (predictions[i] == 1))
	ans = np.sqrt(2*((s+b+br)*np.log(1+float(s)/(b+br))-s))
	if ~np.isfinite(ans):
		return 0
	return ans
	
def precision(ground_truth, predictions, weights=None):
	if weights:
		tp = np.sum(weights[i] for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 1) and (predictions[i] == 1))
		fp = np.sum(weights[i] for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 0) and (predictions[i] == 1))
	else:
		tp = np.sum(1 for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 1) and (predictions[i] == 1))
		fp = np.sum(1 for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 0) and (predictions[i] == 1))
	return float(tp)/(tp+fp)

def recall(ground_truth, predictions, weights=None):
	if weights:
		tp = np.sum(weights[i] for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 1) and (predictions[i] == 1))
		fn = np.sum(weights[i] for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 1) and (predictions[i] == 0))
	else:
		tp = np.sum(1 for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 1) and (predictions[i] == 1))
		fn = np.sum(1 for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 1) and (predictions[i] == 0))
	return float(tp)/(tp+fn)
	
def fbeta(ground_truth, predictions, weights=None, beta=0.5):
	if weights:
		tp = np.sum(weights[i] for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 1) and (predictions[i] == 1))
		fp = np.sum(weights[i] for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 0) and (predictions[i] == 1))
		fn = np.sum(weights[i] for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 1) and (predictions[i] == 0))
	else:
		tp = np.sum(1 for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 1) and (predictions[i] == 1))
		fp = np.sum(1 for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 0) and (predictions[i] == 1))
		fn = np.sum(1 for i in xrange(len(ground_truth)) \
					if (ground_truth[i] == 1) and (predictions[i] == 0))
	p = float(tp)/(tp+fp)
	r = float(tp)/(tp+fn)
	return (1+beta**2)*p*r/((beta**2)*p+r)	
	
def f1(ground_truth, predictions, weights=None):
	return fbeta(ground_truth, predictions, weights, beta=1.0)


def search_best_score(y, y_pred, w, cutoff_thresholds, func=asm):
	''' Helper function to find best possible score using `func`
		for score evaluation '''
	best_score = -1
	for ct in cutoff_thresholds:
		y_copy = y_pred.copy()
		cutoff = (1 - ct) * np.max(y_copy)
		for i in xrange(len(y_copy)):
			if y_copy[i] < cutoff:
				y_copy[i] = 0
			else:
				y_copy[i] = 1
		y_copy = y_copy.astype(np.int8)
		curr_score = func(y, y_copy, w)
		if curr_score > best_score:
			best_score = curr_score
			best_ct = ct
	return best_score, best_ct
	
	
