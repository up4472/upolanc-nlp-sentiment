from typing import Dict
from typing import List

from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold
from gensim.models import Word2Vec

from pandas import DataFrame
from typing import Tuple

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import defaultdict

from src.cnn import cnn_predict
from src.cnn import cnn_train
from src.cnn import create_cnn
from src.system import get_random_state

import tensorflow.keras.preprocessing.sequence
import tensorflow.keras.backend
import os.path
import numpy

max_vocab_size = 10000
max_seq_length = 25
embedding_dim = 100

def __cnn_init (dataset : DataFrame, column : str) -> None :
	global max_vocab_size
	global max_seq_length

	max_seq_length = 1 + max(dataset[column].apply(lambda x : len(x.split())))
	max_vocab_size = 500 + len(set(numpy.concatenate(dataset[column].apply(lambda x : x.split()).to_numpy())))

def __cnn_keras (x_train : numpy.ndarray, y_train : numpy.ndarray, x_test : numpy.ndarray, y_test : numpy.ndarray,
				 n_classes : int, epochs : int, history : dict, embedding_file : str, dataset : DataFrame) -> None :

	if embedding_file is None :
		embedding_file = 'res/glove.6B.100d.txt'

	global max_seq_length
	global max_vocab_size
	global embedding_dim

	seq_length = max_seq_length
	vocab_size = max_vocab_size

	# Tokenize vocabulary
	tokenizer = Tokenizer(num_words = vocab_size)
	tokenizer.fit_on_texts(texts = list(x_train))

	x_train = tokenizer.texts_to_sequences(x_train)
	x_test = tokenizer.texts_to_sequences(x_test)

	x_train = tensorflow.keras.preprocessing.sequence.pad_sequences(x_train, maxlen = seq_length)
	x_test = tensorflow.keras.preprocessing.sequence.pad_sequences(x_test, maxlen = seq_length)

	# Update sequence length (safety check)
	seq_length = x_train.shape[1]

	# Compute embedding weight matrix
	vocab_size, embedding_matrix = get_embending_matrix(
		filepath = embedding_file,
		tokenizer = tokenizer,
		vocab_size = vocab_size
	)

	# Clear cache and stuff
	tensorflow.keras.backend.clear_session()

	# Create sequential neural network
	model = create_cnn(
		input_dim = vocab_size,
		output_dim = embedding_dim,
		input_size = seq_length,
		output_size = n_classes,
		embedding_matrix = embedding_matrix
	)

	# Train the model
	_ = cnn_train(model = model, xdata = x_train, ydata = y_train, epochs = epochs, batch_size = 128)

	# Predict and evalute the model
	results = cnn_predict(model = model, xdata = x_test, ydata = y_test, batch_size = 128)

	# Save the results
	history['test_accuracy'].append(results['accuracy_score'])
	history['test_precision'].append(results['precision'])
	history['test_recall'].append(results['recall'])
	history['test_f1_score'].append(results['f1_score'])
	history['test_brier_score'].append(results['brier_score'])

def cnn_keras_defsplit (dataset : DataFrame, epochs : int, embedding_file : str = None) -> Dict[str, List[float]] :
	__cnn_init(dataset = dataset, column = 'bert_text')

	n = dataset['target'].nunique()
	x = dataset['bert_text'].to_numpy()
	y = dataset['target'].values

	# To categorical target
	y = to_categorical(y, num_classes = n, dtype = int)

	# Default split of 2-1 ratio
	x_train, x_test, y_train, y_test = train_test_split(x, y,
		test_size = 0.33,
		stratify = y,
		random_state = get_random_state()
	)

	# Empty list for results
	history = defaultdict(list)

	# Run cnn keras procedure
	__cnn_keras(
		x_train = x_train,
		y_train = y_train,
		x_test = x_test,
		y_test = y_test,
		dataset = dataset,
		n_classes = n,
		epochs = epochs,
		embedding_file = embedding_file,
		history = history
	)

	return {
		'accuracy_score' : history['test_accuracy'],
		'precision' : history['test_precision'],
		'recall' : history['test_recall'],
		'f1_score' : history['test_f1_score'],
		'brier_score' : history['test_brier_score']
	}

def cnn_keras_kfold (dataset : DataFrame, epochs : int, k_fold : int, embedding_file : str = None) -> Dict[str, List[float]] :
	__cnn_init(dataset = dataset, column = 'bert_text')

	n = dataset['target'].nunique()
	x = dataset['bert_text'].to_numpy()
	y = dataset['target'].values

	# To categorical target
	z = to_categorical(y, num_classes = n, dtype = int)

	# Define KFold
	kfold = StratifiedKFold(n_splits = k_fold, shuffle = True, random_state = get_random_state())

	# Empty list for results
	history = defaultdict(list)

	# KFold loop
	for train, test in kfold.split(x, y) :
		# Run cnn keras procedure
		__cnn_keras(
			x_train = x[train],
			y_train = z[train],
			x_test = x[test],
			y_test = z[test],
			dataset = dataset,
			n_classes = n,
			epochs = epochs,
			embedding_file = embedding_file,
			history = history
		)

	return {
		'accuracy_score' : history['test_accuracy'],
		'precision' : history['test_precision'],
		'recall' : history['test_recall'],
		'f1_score' : history['test_f1_score'],
		'brier_score' : history['test_brier_score']
	}

def get_word2vec_matrix (filepath : str, dataset : DataFrame, tokenizer : Tokenizer, vocab_size : int) -> Tuple[int, numpy.ndarray] :
	global embedding_dim

	if os.path.exists(filepath) :
		model = Word2Vec.load(filepath)
	else :
		tokens = list(dataset['text'].apply(lambda x : x.split()).values)

		model = Word2Vec(
			sentences = tokens,
			sg = 1,
			vector_size = embedding_dim,
			window = 5,
			workers = 4,
			min_count = 1,
			epochs = 20
		)

		model.save(filepath)

	word_index = tokenizer.word_index
	n_words = min(vocab_size, len(word_index) + 1)

	embedding_index = {}
	embedding_matrix = numpy.zeros((n_words, embedding_dim))

	for word in list(model.wv.index_to_key) :
		embedding_index[word] = model.wv[word]

	for word, index in tokenizer.word_index.items() :
		if index >= n_words :
			continue

		embedding_vector = embedding_index.get(word)

		if embedding_vector is not None :
			embedding_matrix[index] = embedding_vector

	return len(model.wv), embedding_matrix

def get_embending_matrix (filepath : str, tokenizer : Tokenizer, vocab_size : int) -> Tuple[int, numpy.ndarray] :
	def get_coef (item : str, *array) :
		return item, numpy.asarray(array, dtype = numpy.float32)

	with open(filepath, mode = 'r', encoding = 'utf8') as file :
		vocabulary = [get_coef(*line.rstrip().rsplit()) for line in file]
		# Check if len(word[1]) > 5, basically makes word2vec compatible with this glove read format, since word2vec
		# file starts with a word count and vector size.
		embedding_index = dict(word for word in vocabulary if len(word[1]) > 5)

	global embedding_dim

	for k, v in embedding_index.items() :
		embedding_dim = len(v)
		break

	word_index = tokenizer.word_index
	n_words = min(vocab_size, len(word_index) + 1)
	embeddings = numpy.stack(embedding_index.values())

	embedding_matrix = numpy.random.normal(embeddings.mean(), embeddings.std(), (n_words, embedding_dim))

	for word, index in word_index.items() :
		if index >= vocab_size :
			continue

		embedding_vector = embedding_index.get(word)

		if embedding_vector is not None :
			embedding_matrix[index] = embedding_vector

	vocab_size = embedding_matrix.shape[0]

	return vocab_size, embedding_matrix
