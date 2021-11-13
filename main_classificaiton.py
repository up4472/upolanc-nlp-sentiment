from pandas import DataFrame

from src.dataset import compute_pos_neg
from src.dataset import encode_target
from src.feature import vader_features
from src.feature import vader_features_ext
from src.models import evaluate_classification
from src.models import train_model

import logging
import numpy

def __report (name : str, results : dict) -> None :
	mean_acc = numpy.mean(results['accuracy_score'])
	std_acc = numpy.std(results['accuracy_score'])

	mean_bs = numpy.mean(results['brier_score'])
	std_bs = numpy.std(results['brier_score'])

	logging.info(f'[{name:24}] Accuracy score : {mean_acc:.5f} \u00B1 {std_acc:.5f}')
	logging.info(f'[{name:24}] Brier score    : {mean_bs:.5f} \u00B1 {std_bs:.5f}\n')

def __class_majority (dataset : DataFrame) -> None :
	majority = numpy.empty(shape = (len(dataset), 1))
	majority.fill(dataset['target'].mode()[0])

	probabilty = numpy.empty(shape = (len(dataset), 3))
	probabilty[:, dataset['target'].mode()[0]] = 1

	result = evaluate_classification(
		ytrue = dataset['target'].to_numpy(),
		ypred = majority,
		yprob = probabilty
	)

	__report('Majority classifier', result)

def __class_vader (dataset : DataFrame) -> None :
	result = evaluate_classification(
		ytrue = dataset['target'].to_numpy(),
		ypred = dataset['vader_prediction'].to_numpy()
	)

	__report('Vader if-else', result)

def __class_vader_ml (dataset : DataFrame, model_name : str) -> None :
	xdata, ydata = vader_features(dataset = dataset)

	results = train_model(xdata = xdata, ydata = ydata, model_name = model_name, k_fold = 10)

	__report(f'Vader Features [{model_name}]', results)

def __class_vader_ml_ext (dataset : DataFrame, model_name : str) -> None :
	xdata, ydata = vader_features_ext(dataset = dataset)

	results = train_model(xdata = xdata, ydata = ydata, model_name = model_name, k_fold = 10)

	__report(f'Vader Features Ext [{model_name}]', results)

def main_classification (dataset : DataFrame, pos_words : set, neg_words : set) -> None :
	logging.info('Creating classification target...')
	dataset, encoder = encode_target(dataset = dataset, target = 'target', column = 'vader_prediction')

	logging.info('Calculating number of positive and negative words....\n')
	dataset = compute_pos_neg(dataset = dataset, column = 'text', pos_words = pos_words, neg_words = neg_words)

	logging.info('Adding a constant to all columns that have negative values....\n')
	dataset['vader_compound'] = dataset['vader_compound'].apply(lambda x : 1 + x)

	logging.debug('Dataset header:\n' + str(dataset.head()) + '\n')

	# MAJORITY VOTING
	__class_majority(dataset = dataset)

	# VADER
	__class_vader(dataset = dataset)

	for name in ['MNB', 'GNB', 'DT', 'KNN', 'RF'] :
		__class_vader_ml(dataset = dataset, model_name = name)
		__class_vader_ml_ext(dataset = dataset, model_name = name)

	# BERT
	...
