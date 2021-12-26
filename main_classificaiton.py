from pandas import DataFrame
from logging import Logger

from src.models import MODELS
from src.models import NAMES

from src.dataset import compute_pos_neg
from src.dataset import encode_target
from src.eval import classification_to_list
from src.eval import classification_to_list_foreach
from src.eval import evaluate_classification
from src.feature import vader_features
from src.feature import vader_features_ext
from src.models import models_kfold
from src.bert import bert_defsplit
from src.cnn_keras import cnn_keras_defsplit
from src.cnn_keras import cnn_keras_kfold
from src.system import init_device

import numpy

def get_ending (value: int) -> str :
	if value == 1 : return 'st'
	if value == 2 : return 'nd'
	if value == 3 : return 'rd'
	return 'th'

def __append_report (results : dict, name : str, feature : str, report : list, foreach : bool = False) -> None :
	items = [name, feature]

	for result in classification_to_list(results = results) :
		items.append(result)

	report.append(items)

	if foreach :
		for index, history in enumerate(classification_to_list_foreach(results = results)) :
			items = [f'{name} ({index + 1:2d}-{get_ending(index + 1)} fold)', feature]

			for result in history :
				items.append(result)

			report.append(items)

def __majority (dataset : DataFrame, report : list) :
	majority = numpy.empty(shape = (len(dataset), 1))
	majority.fill(dataset['target'].mode()[0])

	probabilty = numpy.empty(shape = (len(dataset), 3))
	probabilty[:, dataset['target'].mode()[0]] = 1

	results = evaluate_classification(
		ytrue = dataset['target'].to_numpy(),
		ypred = majority,
		yprob = probabilty
	)

	__append_report(results = results, name = 'Majority Classifier', feature = 'n/a', report = report)

def __vader_v0 (dataset : DataFrame, report : list) -> None :
	results = evaluate_classification(
		ytrue = dataset['target'].to_numpy(),
		ypred = dataset['vader_prediction'].to_numpy()
	)

	__append_report(results = results, name = 'Rule Based', feature = 'VADER', report = report)

def __vader_v1 (dataset : DataFrame, name : str, report : list, k_fold : int = 10) -> None :
	xdata, ydata = vader_features(dataset = dataset)

	results = models_kfold(xdata = xdata, ydata = ydata, model_name = name, k_fold = k_fold)

	__append_report(results = results, name = f'{NAMES[name]}', feature = 'VADER', report = report)

def __vader_v2 (dataset : DataFrame, name : str, report : list, k_fold : int = 10) -> None :
	xdata, ydata = vader_features_ext(dataset = dataset)

	results = models_kfold(xdata = xdata, ydata = ydata, model_name = name, k_fold = k_fold)

	__append_report(results = results, name = f'{NAMES[name]}', feature = 'custom + VADER', report = report)

def __bert (dataset : DataFrame, epochs : int, save_model : bool, report : list) -> None :
	dataset = dataset[['target', 'bert_text']]

	results = bert_defsplit(dataset = dataset, epochs = epochs, save_model = save_model)

	__append_report(results = results, name = 'BERT', feature = 'BERT', report = report, foreach = False)

def __cnn_keras_defsplit (dataset : DataFrame, epochs : int, report : list, embedding_type : str) -> None :
	dataset = dataset[['target', 'text', 'bert_text']]

	results = cnn_keras_defsplit(dataset = dataset, epochs = epochs, embedding_type = embedding_type.lower())

	__append_report(results = results, name = 'CNN Keras', feature = embedding_type, report = report)

def __cnn_keras_kfold (dataset : DataFrame, epochs : int, report : list, embedding_type : str, k_fold : int = 10, ) -> None :
	dataset = dataset[['target', 'text', 'bert_text']]

	results = cnn_keras_kfold(dataset = dataset, epochs = epochs, k_fold = k_fold, embedding_type = embedding_type.lower())

	__append_report(results = results, name = 'CNN Keras', feature = embedding_type, report = report, foreach = False)

def main_classification (dataset : DataFrame, pos_words : set, neg_words : set, logger : Logger) -> None :
	logger.info('Creating classification target...')
	dataset, encoder = encode_target(dataset = dataset, target = 'target', column = 'vader_prediction')

	logger.info('Calculating number of positive and negative words....\n')
	dataset = compute_pos_neg(dataset = dataset, column = 'tokens', pos_words = pos_words, neg_words = neg_words)

	logger.info('Adding a constant to all columns that have negative values....\n')
	dataset['vader_compound'] = dataset['vader_compound'].apply(lambda x : 1 + x)

	logger.debug('Dataset header:\n' + str(dataset.head()) + '\n')

	# INIT DEVICE
	init_device(logger = logger)

	# REPORTS
	columns = ['model', 'features', 'accuracy', 'precision', 'recall', 'f1_score', 'brier_score']
	reports = list()

	# MAJORITY CLASSIFIER
	logger.info('Processing majority classifier...')
	__majority(dataset = dataset, report = reports)

	# VADER + MACHINE LEARNING
	logger.info('Processing VADER_v0 model...')
	__vader_v0(dataset = dataset, report = reports)

	for name in MODELS :
		logger.info(f'Processing VADER_v1 model : {name}...')
		__vader_v1(dataset = dataset, name = name, report = reports, k_fold = 10)

		logger.info(f'Processing VADER_v2 model : {name}...')
		__vader_v2(dataset = dataset, name = name, report = reports, k_fold = 10)

	# BERT
	logger.info('Processing BERT model...')
	__bert(dataset = dataset, epochs = 5, save_model = True, report = reports)

	# CNN
	logger.info('Processing CNN model : Word2vec...')
	__cnn_keras_kfold(dataset = dataset, epochs = 5, report = reports, k_fold = 10, embedding_type = 'Word2vec')

	logger.info('Processing CNN model : GloVe...\n')
	__cnn_keras_kfold(dataset = dataset, epochs = 5, report = reports, k_fold = 10, embedding_type = 'GloVe')

	# PREPARE FINAL REPORT DATAFRAME
	report = DataFrame(reports, columns = columns)

	logger.info('Final report :\n' + str(report))
