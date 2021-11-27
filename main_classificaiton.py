from pandas import DataFrame
from logging import Logger

from src.dataset import compute_pos_neg
from src.dataset import encode_target
from src.feature import vader_features
from src.feature import vader_features_ext
from src.models import evaluate_classification
from src.models import models_method
from src.bert import bert_method

import numpy

def __report (name : str, results : dict, logger : Logger) -> None :
	mean_acc = numpy.mean(results['accuracy_score'])
	std_acc = numpy.std(results['accuracy_score'])

	mean_bs = numpy.mean(results['brier_score'])
	std_bs = numpy.std(results['brier_score'])

	mean_pre = numpy.mean(results['precision'])
	std_pre = numpy.std(results['precision'])

	mean_rec = numpy.mean(results['recall'])
	std_rec = numpy.std(results['recall'])

	mean_f1 = numpy.mean(results['f1_score'])
	std_f1 = numpy.std(results['f1_score'])

	logger.info(f'[{name:24}] Accuracy    : {mean_acc:.5f} \u00B1 {std_acc:.5f}')
	logger.info(f'[{name:24}] Precision   : {mean_pre:.5f} \u00B1 {std_pre:.5f}')
	logger.info(f'[{name:24}] Recall      : {mean_rec:.5f} \u00B1 {std_rec:.5f}')
	logger.info(f'[{name:24}] F1 score    : {mean_f1:.5f} \u00B1 {std_f1:.5f}')
	logger.info(f'[{name:24}] Brier score : {mean_bs:.5f} \u00B1 {std_bs:.5f}\n')

def __class_majority (dataset : DataFrame, logger : Logger) -> None :
	majority = numpy.empty(shape = (len(dataset), 1))
	majority.fill(dataset['target'].mode()[0])

	probabilty = numpy.empty(shape = (len(dataset), 3))
	probabilty[:, dataset['target'].mode()[0]] = 1

	results = evaluate_classification(
		ytrue = dataset['target'].to_numpy(),
		ypred = majority,
		yprob = probabilty
	)

	__report(
		name = f'Majority classifier',
		results = results,
		logger = logger
	)

def __class_vader (dataset : DataFrame, logger : Logger) -> None :
	results = evaluate_classification(
		ytrue = dataset['target'].to_numpy(),
		ypred = dataset['vader_prediction'].to_numpy()
	)

	__report(
		name = f'Vader [compound score]',
		results = results,
		logger = logger
	)

def __class_vader_ml (dataset : DataFrame, model_name : str, logger : Logger) -> None :
	xdata, ydata = vader_features(dataset = dataset)

	results = models_method(xdata = xdata, ydata = ydata, model_name = model_name, k_fold = 10)

	__report(
		name = f'Vader Features [{model_name}]',
		results = results,
		logger = logger
	)

def __class_vader_ml_ext (dataset : DataFrame, model_name : str, logger : Logger) -> None :
	xdata, ydata = vader_features_ext(dataset = dataset)

	results = models_method(xdata = xdata, ydata = ydata, model_name = model_name, k_fold = 10)

	__report(
		name = f'Vader Features Ext [{model_name}]',
		results = results,
		logger = logger
	)

def __class_bert (dataset : DataFrame, epochs : int, logger : Logger) -> None :
	results = bert_method(dataset = dataset, epochs = epochs, save_model = True)

	name = f'BERT model [epochs = {epochs}]'

	neg_mean = numpy.mean(results['accuracy_per_class'][0])
	neg_std = numpy.std(results['accuracy_per_class'][0])

	neu_mean = numpy.mean(results['accuracy_per_class'][1])
	neu_std = numpy.std(results['accuracy_per_class'][1])

	pos_mean = numpy.mean(results['accuracy_per_class'][2])
	pos_std = numpy.std(results['accuracy_per_class'][2])

	logger.info(f'[{name:24}] Accuracy [0]: {neg_mean:.5f} \u00B1 {neg_std:.5f}')
	logger.info(f'[{name:24}] Accuracy [1]: {neu_mean:.5f} \u00B1 {neu_std:.5f}')
	logger.info(f'[{name:24}] Accuracy [2]: {pos_mean:.5f} \u00B1 {pos_std:.5f}')

	__report(
		name = name,
		results = results,
		logger = logger
	)

def main_classification (dataset : DataFrame, pos_words : set, neg_words : set, logger : Logger) -> None :
	logger.info('Creating classification target...')
	dataset, encoder = encode_target(dataset = dataset, target = 'target', column = 'vader_prediction')

	logger.info('Calculating number of positive and negative words....\n')
	dataset = compute_pos_neg(dataset = dataset, column = 'tokens', pos_words = pos_words, neg_words = neg_words)

	logger.info('Adding a constant to all columns that have negative values....\n')
	dataset['vader_compound'] = dataset['vader_compound'].apply(lambda x : 1 + x)

	logger.debug('Dataset header:\n' + str(dataset.head()) + '\n')

	# MAJORITY VOTING
	__class_majority(dataset = dataset, logger = logger)

	# VADER + MACHINE LEARNING
	__class_vader(dataset = dataset, logger = logger)

	for name in ['MNB', 'GNB', 'DT', 'KNN', 'RF', 'MV'] :
		__class_vader_ml(dataset = dataset, model_name = name, logger = logger)
		__class_vader_ml_ext(dataset = dataset, model_name = name, logger = logger)

	# BERT
	for epochs in [1] :
		__class_bert(dataset = dataset, epochs = epochs, logger = logger)
