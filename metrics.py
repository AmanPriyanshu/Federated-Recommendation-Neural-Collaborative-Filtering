import numpy as np

def hit_ratio(y, pred, N=10):
	mask = np.zeros_like(y)
	mask[y>0] = 1
	pred_masked = pred*mask
	best_index = np.argmax(y)
	pred_masked_indexes = np.argsort(pred_masked)[::-1][:N]
	if best_index in pred_masked_indexes:
		return 1
	else:
		return 0

def ndcg(y, pred, N=10):
	actual_recommendation_best_10indexes = np.argsort(y)[::-1][:N]
	actual_recommendation_best_10 = y[actual_recommendation_best_10indexes]
	predicted_recommendation_best_10 = pred[actual_recommendation_best_10indexes]
	predicted_recommendation_best_10 = np.around(predicted_recommendation_best_10)
	predicted_recommendation_best_10[predicted_recommendation_best_10<0] = 0
	dcg_numerator = np.power(2, predicted_recommendation_best_10) - 1
	denomimator = np.log2(np.arange(start=2, stop=N+2))
	idcg_numerator = np.power(2, actual_recommendation_best_10) - 1
	dcg = np.sum(dcg_numerator/denomimator)
	idcg = np.sum(idcg_numerator/denomimator)
	if idcg!=0:
		ndcg = dcg/idcg
	else:
		ndcg = 0.0
	return ndcg

def compute_metrics(y, pred, metric_functions=None):
	if metric_functions is None:
		metric_functions = [hit_ratio, ndcg]
	return [fun(y, pred) for fun in metric_functions]