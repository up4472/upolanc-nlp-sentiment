from nltk.corpus import stopwords
from logging import FileHandler
from logging import Formatter
from logging import Logger

from main_classificaiton import main_classification
from src.dataset import compute_polarity
from src.dataset import lemmatize_tokens
from src.dataset import load_csv
from src.dataset import load_json
from src.dataset import clean_text
from src.dataset import sentimental_words
from src.dataset import stem_tokens
from src.dataset import tokenize_text

import logging
import pandas
import string
import nltk

pandas.set_option('display.max_rows', 1000)
pandas.set_option('display.max_columns', 1000)
pandas.set_option('display.width', 1000)

def create_logger (filename : str = None, level : int = logging.DEBUG) -> Logger :
	logger = logging.getLogger()
	handler = FileHandler(filename = filename, mode = 'w', encoding = 'utf-8')

	handler.setFormatter(fmt = Formatter('%(asctime)s - %(name)s : %(levelname)s : %(message)s'))
	logger.setLevel(level = level)
	logger.addHandler(handler)

	return logger
	# 	filename = filename,
	# 	filemode = 'w',
	# 	format = '%(asctime)s - %(name)s : %(levelname)s : %(message)s',
	# 	encoding = 'UTF-8',
	# 	level = level
	# )

def download_resources (logger : Logger) -> None :
	logger.info('Downloading [stopwords] from [nltk]...')
	nltk.download('stopwords', quiet = True)

	logger.info('Downloading [names] from [nltk]...')
	nltk.download('names', quiet = True)

	logger.info('Downloading [wordnet] from [nltk]...')
	nltk.download('wordnet', quiet = True)

	logger.info('Downloading [vader_lexicon] from [nltk]...')
	nltk.download('vader_lexicon', quiet = True)

def main (logger : Logger) -> None :
	logger.info('Loading dataset and configuration files...')

	dataset = load_csv(filename = 'airlines.csv')
	config = load_json(filename = 'airlines.json')

	dataset = dataset[config['columns']]
	dataset = dataset.rename(columns = {'airline_sentiment' : 'target'})

	logger.info('Checking for any missing values in the dataset...\n')
	logger.debug('Missing values:\n' + str(dataset.isnull().sum()) + '\n')
	logger.debug('Dataset header:\n' + str(dataset.head()) + '\n')

	en_stopwords = stopwords.words('english')
	en_punct = string.punctuation

	distribution = dataset['target'].value_counts().to_frame(name = 'count')
	distribution['frequency'] = distribution['count'] / distribution['count'].sum()

	logger.debug('Sentiment distribution:\n' + str(distribution) + '\n')

	logger.info('Cleaning dataset text...')
	dataset = clean_text(dataset = dataset, column = 'text', stopwords = en_stopwords, punct = en_punct)

	logger.info('Calculating polarity scores...')
	dataset = compute_polarity(dataset = dataset, column = 'text')

	logger.info('Tokenizing dataset text...')
	dataset = tokenize_text(dataset = dataset, column = 'text')

	logger.info('Stemming dataset text...')
	dataset = stem_tokens(dataset = dataset, column = 'text')

	logger.info('Lemmatizing dataset text...')
	dataset = lemmatize_tokens(dataset = dataset, column = 'text')

	logger.debug('Dataset header:\n' + str(dataset.head()) + '\n')

	pos_words, neg_words = sentimental_words(dataset = dataset, column = 'text', target = 'target', top = 100)

	logger.debug('Positive words : ' + str(pos_words))
	logger.debug('Negative words : ' + str(neg_words) + '\n')

	main_classification(dataset = dataset, pos_words = pos_words, neg_words = neg_words, logger = logger)

if __name__ == '__main__' :
	file_logger = create_logger(filename = 'output.log', level = logging.DEBUG)

	download_resources(logger = file_logger)
	main(logger = file_logger)
