from sklearn.preprocessing import LabelBinarizer
from typing import Dict

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import numpy

def evaluate_classification (ytrue : numpy.ndarray, ypred : numpy.ndarray, yprob : numpy.ndarray = None) -> Dict[str, float] :
	_accuracy = accuracy_score(y_true = ytrue, y_pred = ypred)
	_confusion = confusion_matrix(y_true = ytrue, y_pred = ypred)
	_metrics = precision_recall_fscore_support(y_true = ytrue, y_pred = ypred, average = 'weighted', zero_division = 0)
	_brier = numpy.nan

	_precision = _metrics[0]
	_recall = _metrics[1]
	_f1score = _metrics[2]

	if yprob is not None :
		lb = LabelBinarizer()

		_brier = numpy.sum(numpy.square(numpy.subtract(lb.fit_transform(ytrue), yprob))) / len(ytrue)

	return {
		'accuracy_score' : _accuracy,
		'precision' : _precision,
		'recall' : _recall,
		'f1_score' : _f1score,
		'brier_score' : _brier
	}

def classification_to_list (results : dict) -> list :
	acc = f'{numpy.mean(results["accuracy_score"]):.5f} \u00B1 {numpy.std(results["accuracy_score"]):.5f}'
	pre = f'{numpy.mean(results["precision"]):.5f} \u00B1 {numpy.std(results["precision"]):.5f}'
	rec = f'{numpy.mean(results["recall"]):.5f} \u00B1 {numpy.std(results["recall"]):.5f}'
	f1s = f'{numpy.mean(results["f1_score"]):.5f} \u00B1 {numpy.std(results["f1_score"]):.5f}'
	bsc = f'{numpy.mean(results["brier_score"]):.5f} \u00B1 {numpy.std(results["brier_score"]):.5f}'

	return [acc, pre, rec, f1s, bsc]

def classification_to_list_foreach (results : dict) -> list :
	result = []

	for index in range(len(results['accuracy_score'])) :
		acc = f'{numpy.mean(results["accuracy_score"][index]):.5f} \u00B1 {numpy.std(results["accuracy_score"][index]):.5f}'
		pre = f'{numpy.mean(results["precision"][index]):.5f} \u00B1 {numpy.std(results["precision"][index]):.5f}'
		rec = f'{numpy.mean(results["recall"][index]):.5f} \u00B1 {numpy.std(results["recall"][index]):.5f}'
		f1s = f'{numpy.mean(results["f1_score"][index]):.5f} \u00B1 {numpy.std(results["f1_score"][index]):.5f}'
		bsc = f'{numpy.mean(results["brier_score"][index]):.5f} \u00B1 {numpy.std(results["brier_score"][index]):.5f}'

		result.append([acc, pre, rec, f1s, bsc])

	return result
