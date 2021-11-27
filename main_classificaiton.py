from pandas import DataFrame
from logging import Logger

from src.dataset import compute_pos_neg
from src.dataset import encode_target
from src.eval import classification_to_list
from src.feature import vader_features
from src.feature import vader_features_ext
from src.models import evaluate_classification
from src.models import models_kfold
from src.bert import bert_defsplit

import numpy

def __majority (dataset : DataFrame, report : list, names : list) :
	majority = numpy.empty(shape = (len(dataset), 1))
	majority.fill(dataset['target'].mode()[0])

	probabilty = numpy.empty(shape = (len(dataset), 3))
	probabilty[:, dataset['target'].mode()[0]] = 1

	results = evaluate_classification(
		ytrue = dataset['target'].to_numpy(),
		ypred = majority,
		yprob = probabilty
	)

	report.append(classification_to_list(results = results))
	names.append('Majority Classifier')

def __vader_v0 (dataset : DataFrame, report : list, names : list) -> None :
	results = evaluate_classification(
		ytrue = dataset['target'].to_numpy(),
		ypred = dataset['vader_prediction'].to_numpy()
	)

	report.append(classification_to_list(results = results))
	names.append('VADER Classifier (Argmax)')

def __vader_v1 (dataset : DataFrame, name : str, report : list, names : list) -> None :
	xdata, ydata = vader_features(dataset = dataset)

	results = models_kfold(xdata = xdata, ydata = ydata, model_name = name, k_fold = 10)

	report.append(classification_to_list(results = results))
	names.append(f'VADER As Feature ({name})')

def __vader_v2 (dataset : DataFrame, name : str, report : list, names : list) -> None :
	xdata, ydata = vader_features_ext(dataset = dataset)

	results = models_kfold(xdata = xdata, ydata = ydata, model_name = name, k_fold = 10)

	report.append(classification_to_list(results = results))
	names.append(f'VADER As Feature + Custom ({name})')

def __bert_v0 (dataset : DataFrame, epochs : int, report : list, names : list, logger : Logger) -> None :
	results = bert_defsplit(dataset = dataset, epochs = epochs, save_model = True)

	name = f'BERT ({epochs} epochs)'

	neg_mean = numpy.mean(results['accuracy_per_class'][0])
	neg_std = numpy.std(results['accuracy_per_class'][0])

	neu_mean = numpy.mean(results['accuracy_per_class'][1])
	neu_std = numpy.std(results['accuracy_per_class'][1])

	pos_mean = numpy.mean(results['accuracy_per_class'][2])
	pos_std = numpy.std(results['accuracy_per_class'][2])

	logger.info(f'[{name:16}] Accuracy [0]: {neg_mean:.5f} \u00B1 {neg_std:.5f}')
	logger.info(f'[{name:16}] Accuracy [1]: {neu_mean:.5f} \u00B1 {neu_std:.5f}')
	logger.info(f'[{name:16}] Accuracy [2]: {pos_mean:.5f} \u00B1 {pos_std:.5f}\n')

	report.append(classification_to_list(results = results))
	names.append(name)

def main_classification (dataset : DataFrame, pos_words : set, neg_words : set, logger : Logger) -> None :
	logger.info('Creating classification target...')
	dataset, encoder = encode_target(dataset = dataset, target = 'target', column = 'vader_prediction')

	logger.info('Calculating number of positive and negative words....\n')
	dataset = compute_pos_neg(dataset = dataset, column = 'tokens', pos_words = pos_words, neg_words = neg_words)

	logger.info('Adding a constant to all columns that have negative values....\n')
	dataset['vader_compound'] = dataset['vader_compound'].apply(lambda x : 1 + x)

	logger.debug('Dataset header:\n' + str(dataset.head()) + '\n')

	# REPORTS
	columns = ['accuracy', 'precision', 'recall', 'f1_score', 'brier_score']
	reports = list()
	names = list()

	# MAJORITY VOTING
	__majority(dataset = dataset, report = reports, names = names)

	# VADER + MACHINE LEARNING
	__vader_v0(dataset = dataset, report = reports, names = names)

	for name in ['MNB', 'GNB', 'DT', 'KNN', 'RF', 'MV'] :
		__vader_v1(dataset = dataset, name = name, report = reports, names = names)
		__vader_v2(dataset = dataset, name = name, report = reports, names = names)

	# BERT
	for epochs in [1] :
		__bert_v0(dataset = dataset, epochs = epochs, report = reports, names = names, logger = logger)

	# PREPARE FINAL REPORT DATAFRAME
	report = DataFrame(reports, columns = columns, index = names)

	logger.info('Final report :\n' + str(report))
