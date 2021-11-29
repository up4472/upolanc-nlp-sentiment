from pandas import DataFrame
from logging import Logger

from src.dataset import compute_pos_neg
from src.dataset import encode_target
from src.eval import classification_to_list
from src.eval import classification_to_list_foreach
from src.eval import evaluate_classification
from src.feature import vader_features
from src.feature import vader_features_ext
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
	names.append('Majority')

def __vader_v0 (dataset : DataFrame, report : list, names : list) -> None :
	results = evaluate_classification(
		ytrue = dataset['target'].to_numpy(),
		ypred = dataset['vader_prediction'].to_numpy()
	)

	report.append(classification_to_list(results = results))
	names.append('VADER Rule-Based')

def __vader_v1 (dataset : DataFrame, name : str, report : list, names : list) -> None :
	xdata, ydata = vader_features(dataset = dataset)

	results = models_kfold(xdata = xdata, ydata = ydata, model_name = name, k_fold = 10)

	report.append(classification_to_list(results = results))
	names.append(f'VADER Features [{name}]')

def __vader_v2 (dataset : DataFrame, name : str, report : list, names : list) -> None :
	xdata, ydata = vader_features_ext(dataset = dataset)

	results = models_kfold(xdata = xdata, ydata = ydata, model_name = name, k_fold = 10)

	report.append(classification_to_list(results = results))
	names.append(f'VADER + Custom Feartures [{name}]')

def __bert (dataset : DataFrame, epochs : int, report : list, names : list) -> None :
	results = bert_defsplit(dataset = dataset, epochs = epochs, name = 'sentiment', save_model = True)

	def get_ending (value : int) -> str :
		if value == 1 : return 'st'
		if value == 2 : return 'nd'
		if value == 3 : return 'rd'
		return 'th'

	for index, result in enumerate(classification_to_list_foreach(results = results, epochs = epochs)) :
		report.append(result)
		names.append(f'BERT-sent [{index + 1}-{get_ending(index + 1)} epoch]')

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
	print('Running Majority')

	__majority(dataset = dataset, report = reports, names = names)

	# VADER + MACHINE LEARNING
	print('Running VADERs')

	__vader_v0(dataset = dataset, report = reports, names = names)

	for name in ['MNB', 'GNB', 'DT', 'KNN', 'RF', 'MV'] :
		__vader_v1(dataset = dataset, name = name, report = reports, names = names)
		__vader_v2(dataset = dataset, name = name, report = reports, names = names)

	# BERT
	print('Running BERTs')
	__bert(dataset = dataset[['target', 'text']], epochs = 2, report = reports, names = names)

	# PREPARE FINAL REPORT DATAFRAME
	report = DataFrame(reports, columns = columns, index = names)

	logger.info('Final report :\n' + str(report))
