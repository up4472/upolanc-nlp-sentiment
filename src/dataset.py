from sklearn.preprocessing import LabelEncoder
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
from nltk import PorterStemmer
from nltk import FreqDist
from pandas import DataFrame

import pandas
import nltk
import json
import os
import re

def load_csv (filename : str) -> DataFrame :
	filename = os.path.join('res', filename)

	return pandas.read_csv(filename)

def load_json (filename : str) -> dict :
	filename = os.path.join('res', filename)

	with open(filename, 'r') as file :
		config = json.load(fp = file)

	return config

def clean_text (dataset : DataFrame, column : str, punct : str, stopwords : list) -> DataFrame :
	# Expand stopwords with names
	stopwords.extend([word.lower() for word in nltk.corpus.names.words()])

	# Define some private function
	def clean_url (text) :
		return ' '.join(re.sub(r'((www.\S+)|(http?://\S+))', '', text).split())

	def clean_mentions (text) :
		return ' '.join(re.sub(r'((@\S+)|(#\S+))', '', text).split())

	def clean_emoji (text) :
		return ' '.join(re.sub(r'\W+', ' ', text).split())

	def clean_punctuation (text) :
		return text.translate(text.maketrans('', '', punct))

	def clean_numeric (text) :
		return ' '.join(re.sub(r'[0-9]+', '', text).split())

	def clean_stopwords (text) :
		return ' '.join([word for word in str(text).split() if word not in stopwords])

	# To lowercase
	dataset[column] = dataset[column].str.lower()

	# Clean URL
	dataset[column] = dataset[column].apply(lambda x : clean_url(x))

	# Clean mentions
	dataset[column] = dataset[column].apply(lambda x : clean_mentions(x))

	# Make a copy of cleaned up text
	dataset['bert_text'] = dataset[column].copy(deep = True)

	# Clean emojis
	dataset[column] = dataset[column].apply(lambda x : clean_emoji(x))

	# Clean punctuations
	dataset[column] = dataset[column].apply(lambda x : clean_punctuation(x))

	# Clean numeric
	dataset[column] = dataset[column].apply(lambda x : clean_numeric(x))

	# Clean stopwords
	dataset[column] = dataset[column].apply(lambda x : clean_stopwords(x))

	# Make a copy of cleaned up text
	dataset['tokens'] = dataset[column].copy(deep = True)

	# Add number of words
	dataset['word_count'] = dataset[column].apply(lambda x : len(x.split()))

	# Add number of chars
	dataset['char_count'] = dataset[column].apply(lambda x : len(x))

	return dataset

def compute_polarity (dataset : DataFrame, column : str) -> DataFrame :
	# Define VADER (Valence Aware Dictionary and sEntiment Reasoner)
	analyzer = SentimentIntensityAnalyzer()

	# Define private method
	def vader_predict (score: float) -> str :
		if score <= -0.05 : return 'negative'
		if score >= +0.05 : return 'positive'
		return 'neutral'

	# Add VADER polarity scores
	dataset['vader_compound'] = dataset[column].apply(lambda x : analyzer.polarity_scores(x)['compound'])
	dataset['vader_positive'] = dataset[column].apply(lambda x : analyzer.polarity_scores(x)['pos'])
	dataset['vader_negative'] = dataset[column].apply(lambda x : analyzer.polarity_scores(x)['neg'])
	dataset['vader_neutral']  = dataset[column].apply(lambda x : analyzer.polarity_scores(x)['neu'])

	# Add VADER prediction
	dataset['vader_prediction'] = dataset['vader_compound'].apply(lambda x : vader_predict(score = x))

	return dataset

def tokenize_text (dataset : DataFrame, column : str) -> DataFrame :
	# Define regex tokenizer
	tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')

	# Apply tokenizer
	dataset[column] = dataset[column].apply(lambda x : tokenizer.tokenize(x))

	return dataset

def stem_tokens (dataset : DataFrame, column : str) -> DataFrame :
	# Define stemmer
	stemmer = PorterStemmer()

	# Define private function
	def apply_stemmer (text) :
		return [stemmer.stem(word) for word in text]

	# Apply stemmer
	dataset[column] = dataset[column].apply(lambda x : apply_stemmer(x))

	return dataset

def lemmatize_tokens (dataset : DataFrame, column : str) -> DataFrame :
	# Define lemmatizer
	lemmatizer = WordNetLemmatizer()

	# Define private function
	def apply_lemmatizer (text) :
		return [lemmatizer.lemmatize(word) for word in text]

	# Apply lemmatizer
	dataset[column] = dataset[column].apply(lambda x : apply_lemmatizer(x))

	return dataset

def sentimental_words (dataset : DataFrame, column : str, target : str, top : int = 100) -> (set, set) :
	pos_texts = dataset[dataset[target] == 'positive'][column]
	neg_texts = dataset[dataset[target] == 'negative'][column]

	pos_words = list()
	neg_words = list()

	for words in pos_texts :
		pos_words = pos_words + words
	for words in neg_texts :
		neg_words = neg_words + words

	pos_words = FreqDist(pos_words)
	neg_words = FreqDist(neg_words)

	common = set(pos_words).intersection(neg_words)

	for word in common :
		del pos_words[word]
		del neg_words[word]

	pos = {word for word, count in pos_words.most_common(top)}
	neg = {word for word, count in neg_words.most_common(top)}

	return pos, neg

def encode_label (dataset : DataFrame, column : str) -> (DataFrame, LabelEncoder) :
	# Define label encoder
	encoder = LabelEncoder()

	# Fit encoder
	encoder = encoder.fit(dataset[column].unique())

	# Encode data
	dataset[column] = encoder.transform(dataset[column])

	return dataset, encoder

def encode_target (dataset : DataFrame, target : str, column : str = None) -> (DataFrame, LabelEncoder) :
	dataset, encoder = encode_label(dataset = dataset, column = target)

	# Transform column (vader_prediction)
	if column is not None :
		dataset[column] = encoder.transform(dataset[column])

	return dataset, encoder

def compute_pos_neg (dataset : DataFrame, column : str, pos_words : set, neg_words : set) -> DataFrame :
	# Private count methods
	def count_pos_words (words) :
		return sum([1 for word in words if word in pos_words])

	def count_neg_words (words) :
		return sum([1 for word in words if word in neg_words])

	# Count occurances
	dataset['pos_count'] = dataset[column].apply(lambda x : count_pos_words(x))
	dataset['neg_count'] = dataset[column].apply(lambda x : count_neg_words(x))

	return dataset
