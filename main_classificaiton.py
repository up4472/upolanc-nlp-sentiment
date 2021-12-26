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
import time

LOG_FOREACH = False		# logs metrics for each k when using kfold
DEFAULT_KFOLD = 10		# defaulf kfold
BERT_EPOCHS = 5			# number of training epochs of bert
CNN_EPOCHS = 5			# number of training epochs of cnn keras

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

def __majority (dataset : DataFrame, report : list) -> int :
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

	return 0

def __vader_rule_based (dataset : DataFrame, report : list) -> int :
	results = evaluate_classification(
		ytrue = dataset['target'].to_numpy(),
		ypred = dataset['vader_prediction'].to_numpy()
	)

	__append_report(results = results, name = 'Rule Based', feature = 'VADER', report = report)

	return 0

def __vader_core (dataset : DataFrame, name : str, report : list) -> float :
	xdata, ydata = vader_features(dataset = dataset)

	s = time.perf_counter_ns()
	results = models_kfold(xdata = xdata, ydata = ydata, model_name = name, k_fold = DEFAULT_KFOLD)
	e = time.perf_counter_ns()

	__append_report(results = results, name = f'{NAMES[name]}', feature = 'VADER', report = report, foreach = LOG_FOREACH)

	return (e - s) / 1e+9


def __vader_extended (dataset : DataFrame, name : str, report : list) -> float :
	xdata, ydata = vader_features_ext(dataset = dataset)

	s = time.perf_counter_ns()
	results = models_kfold(xdata = xdata, ydata = ydata, model_name = name, k_fold = DEFAULT_KFOLD)
	e = time.perf_counter_ns()

	__append_report(results = results, name = f'{NAMES[name]}', feature = 'custom + VADER', report = report, foreach = LOG_FOREACH)

	return (e - s) / 1e+9

def __bert (dataset : DataFrame, epochs : int, save_model : bool, report : list) -> float :
	dataset = dataset[['target', 'bert_text']]

	s = time.perf_counter_ns()
	results = bert_defsplit(dataset = dataset, epochs = epochs, save_model = save_model)
	e = time.perf_counter_ns()

	__append_report(results = results, name = 'BERT', feature = 'BERT', report = report, foreach = LOG_FOREACH)

	return (e - s) / 1e+9

def __cnn_keras_defsplit (dataset : DataFrame, epochs : int, report : list, embedding_file : str) -> float :
	dataset = dataset[['target', 'text', 'bert_text']]

	s = time.perf_counter_ns()
	results = cnn_keras_defsplit(dataset = dataset, epochs = epochs, embedding_file = embedding_file)
	e = time.perf_counter_ns()

	__append_report(results = results, name = 'CNN Keras', feature = embedding_file, report = report)

	return (e - s) / 1e+9

def __cnn_keras_kfold (dataset : DataFrame, epochs : int, report : list, embedding_file : str) -> float :
	dataset = dataset[['target', 'text', 'bert_text']]

	s = time.perf_counter_ns()
	results = cnn_keras_kfold(dataset = dataset, epochs = epochs, embedding_file = embedding_file, k_fold = DEFAULT_KFOLD)
	e = time.perf_counter_ns()

	__append_report(results = results, name = 'CNN Keras', feature = embedding_file, report = report, foreach = LOG_FOREACH)

	return (e - s) / 1e+9

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

	def log_info (model_name : str, feature_name : str, seconds : float) -> None :
		logger.info(f'Processed [{model_name:10s}] model with [{feature_name:17s}] features in [{seconds:12.2f}] seconds ...')

	# MAJORITY CLASSIFIER
	runtime = __majority(dataset = dataset, report = reports)
	log_info(model_name = 'Majority', feature_name = '-', seconds = runtime)

	# VADER
	runtime = __vader_rule_based(dataset = dataset, report = reports)
	log_info(model_name = 'Rule Based', feature_name = 'VADER', seconds = runtime)

	# VADER + MACHINE LEARNING
	for name in MODELS :
		runtime = __vader_core(dataset = dataset, name = name, report = reports)
		log_info(model_name = name, feature_name = 'VADER', seconds = runtime)

	for name in MODELS :
		runtime = __vader_extended(dataset = dataset, name = name, report = reports)
		log_info(model_name = name, feature_name = 'VADER + custom', seconds = runtime)

	# BERT
	# __bert(dataset = dataset, epochs = BERT_EPOCHS, save_model = True, report = reports)
	# log_info(model_name = 'BERT', feature_name = 'BERT', seconds = runtime)

	# CNN + WORD2VEC | GLOVE
	runtime = __cnn_keras_kfold(dataset = dataset, epochs = CNN_EPOCHS, report = reports, embedding_file = 'res/glove.6B.100d.txt')
	log_info(model_name = 'CNN Keras', feature_name = 'glove.6B.100d', seconds = runtime)

	runtime = __cnn_keras_kfold(dataset = dataset, epochs = CNN_EPOCHS, report = reports, embedding_file = 'res/glove.6B.200d.txt')
	log_info(model_name = 'CNN Keras', feature_name = 'glove.6B.200d', seconds = runtime)

	runtime = __cnn_keras_kfold(dataset = dataset, epochs = CNN_EPOCHS, report = reports, embedding_file = 'res/word2vec.2M.300d.txt')
	log_info(model_name = 'CNN Keras', feature_name = 'word2vec.2M.300d', seconds = runtime)

	# PREPARE FINAL REPORT DATAFRAME
	report = DataFrame(reports, columns = columns)

	logger.info('Final report :\n' + str(report))
