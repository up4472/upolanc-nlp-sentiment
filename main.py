from nltk.corpus import stopwords

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

def create_logger (filename : str = None, level : int = logging.DEBUG) -> None :
	logging.basicConfig(
		filename = filename,
		format = '%(asctime)s - %(name)s : %(levelname)s : %(message)s',
		level = level
	)

def download_resources () -> None :
	logging.info('Downloading [stopwords] from [nltk]...')
	nltk.download('stopwords', quiet = True)

	logging.info('Downloading [names] from [nltk]...')
	nltk.download('names', quiet = True)

	logging.info('Downloading [wordnet] from [nltk]...')
	nltk.download('wordnet', quiet = True)

	logging.info('Downloading [vader_lexicon] from [nltk]...')
	nltk.download('vader_lexicon', quiet = True)

def main () -> None :
	logging.info('Loading dataset and configuration files...')

	dataset = load_csv(filename = 'airlines.csv')
	config = load_json(filename = 'airlines.json')

	dataset = dataset[config['columns']]
	dataset = dataset.rename(columns = {'airline_sentiment' : 'target'})

	logging.info('Checking for any missing values in the dataset...\n')
	logging.debug('Missing values:\n' + str(dataset.isnull().sum()) + '\n')
	logging.debug('Dataset header:\n' + str(dataset.head()) + '\n')

	en_stopwords = stopwords.words('english')
	en_punct = string.punctuation

	distribution = dataset['target'].value_counts().to_frame(name = 'count')
	distribution['frequency'] = distribution['count'] / distribution['count'].sum()

	logging.debug('Sentiment distribution:\n' + str(distribution) + '\n')

	logging.info('Cleaning dataset text...')
	dataset = clean_text(dataset = dataset, column = 'text', stopwords = en_stopwords, punct = en_punct)

	logging.info('Calculating polarity scores...')
	dataset = compute_polarity(dataset = dataset, column = 'text')

	logging.info('Tokenizing dataset text...')
	dataset = tokenize_text(dataset = dataset, column = 'text')

	logging.info('Stemming dataset text...')
	dataset = stem_tokens(dataset = dataset, column = 'text')

	logging.info('Lemmatizing dataset text...')
	dataset = lemmatize_tokens(dataset = dataset, column = 'text')

	logging.debug('Dataset header:\n' + str(dataset.head()) + '\n')

	pos_words, neg_words = sentimental_words(dataset = dataset, column = 'text', target = 'target', top = 100)

	logging.debug('Positive words : ' + str(pos_words))
	logging.debug('Negative words : ' + str(neg_words) + '\n')

	main_classification(dataset = dataset, pos_words = pos_words, neg_words = neg_words)

if __name__ == '__main__' :
	create_logger(filename = None, level = logging.DEBUG)
	download_resources()
	main()
