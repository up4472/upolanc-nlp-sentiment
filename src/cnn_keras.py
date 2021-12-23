from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History
from pandas import DataFrame
from typing import Tuple
from typing import Dict

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import defaultdict

from src.system import get_random_state
from src.eval import evaluate_classification

import tensorflow.keras.preprocessing.sequence
import tensorflow.keras.backend
import numpy

MAX_VOCAB_WORDS = 20000
EMBEDDING_DIM = 100
BATCH_SIZE = 128
MAX_SEQUENCE_LEN = 50

def create_cnn (input_dim : int, output_dim : int, input_size : int, output_size : int, embedding_matrix : numpy.ndarray = None) -> Sequential :
	model = Sequential()

	if embedding_matrix is None :
		model.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_size))
	else :
		model.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_size, weights = [embedding_matrix]))

	model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
	model.add(MaxPooling1D(pool_size = 2))

	model.add(Conv1D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
	model.add(MaxPooling1D(pool_size = 2))

	model.add(Conv1D(filters = 128, kernel_size = 5, padding = 'same', activation = 'relu'))
	model.add(MaxPooling1D(pool_size = 2))

	model.add(Flatten())
	model.add(Dense(units = 128, activation = 'relu'))
	model.add(Dense(units = output_size, activation = 'softmax'))

	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	return model

def cnn_keras_train (model : Sequential, xdata : numpy.ndarray, ydata : numpy.ndarray, epochs : int) -> History :
	history = model.fit(xdata, ydata,
		batch_size = BATCH_SIZE,
		epochs = epochs,
		validation_split = 0.1
	)

	return history

def cnn_keras_predict (model : Sequential, xdata : numpy.ndarray, ydata : numpy.ndarray) -> Dict[str, float] :
	yprob = model.predict(xdata, batch_size = BATCH_SIZE)

	ytrue = numpy.argmax(ydata, axis = 1).flatten()
	ypred = numpy.argmax(yprob, axis = 1).flatten()

	return evaluate_classification(ytrue = ytrue, ypred = ypred, yprob = yprob)

def __cnn_keras (x_train : numpy.ndarray, y_train : numpy.ndarray, x_test : numpy.ndarray, y_test : numpy.ndarray,
				 n_classes : int, epochs : int, history : dict, embedding_file : str) -> None :
	sequence_length = MAX_SEQUENCE_LEN
	vocab_size = MAX_VOCAB_WORDS

	# Tokenize vocabulary
	tokenizer = Tokenizer(num_words = vocab_size)
	tokenizer.fit_on_texts(texts = list(x_train))

	x_train = tokenizer.texts_to_sequences(x_train)
	x_test = tokenizer.texts_to_sequences(x_test)

	x_train = tensorflow.keras.preprocessing.sequence.pad_sequences(x_train, maxlen = sequence_length)
	x_test = tensorflow.keras.preprocessing.sequence.pad_sequences(x_test, maxlen = sequence_length)

	# Update sequence length (safety check)
	sequence_length = x_train.shape[1]

	# Compute embedding weight matrix
	vocab_size, matrix = get_embedding_matrix(
		filepath = embedding_file,
		tokenizer = tokenizer,
		vocab_size = vocab_size
	)

	# Clear cache and stuff
	tensorflow.keras.backend.clear_session()

	# Create sequential neural network
	model = create_cnn(
		input_dim = vocab_size,
		output_dim = EMBEDDING_DIM,
		input_size = sequence_length,
		output_size = n_classes,
		embedding_matrix = matrix
	)

	# Train the model
	_ = cnn_keras_train(model = model, xdata = x_train, ydata = y_train, epochs = epochs)

	# Predict and evalute the model
	results = cnn_keras_predict(model = model, xdata = x_test, ydata = y_test)

	# Save the results
	history['test_accuracy'].append(results['accuracy_score'])
	history['test_precision'].append(results['precision'])
	history['test_recall'].append(results['recall'])
	history['test_f1_score'].append(results['f1_score'])
	history['test_brier_score'].append(results['brier_score'])

def cnn_keras_defsplit (dataset : DataFrame, epochs : int, embedding_file : str = None) -> dict :
	if embedding_file is None :
		embedding_file = 'res/glove.6B.100d.txt'

	n = dataset['target'].nunique()
	x = dataset['text'].to_numpy()
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

def cnn_keras_kfold (dataset : DataFrame, epochs : int, k_fold : int, embedding_file : str = None) :
	if embedding_file is None : embedding_file = 'res/glove.6B.100d.txt'

	n = dataset['target'].nunique()
	x = dataset['text'].to_numpy()
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

def get_embedding_matrix (filepath : str, tokenizer : Tokenizer, vocab_size : int) -> Tuple[int, numpy.ndarray] :
	def get_coef (item : str, *array) :
		return item, numpy.asarray(array, dtype = numpy.float32)

	with open(filepath, mode = 'r', encoding = 'utf8') as file :
		embedding_index = dict(get_coef(*line.rstrip().rsplit()) for line in file)

	word_index = tokenizer.word_index
	n_words = min(vocab_size, len(word_index) + 1)
	embeddings = numpy.stack(embedding_index.values())

	embedding_matrix = numpy.random.normal(embeddings.mean(), embeddings.std(), (n_words, EMBEDDING_DIM))

	for word, index in word_index.items() :
		if index >= vocab_size :
			continue

		embedding_vector = embedding_index.get(word)

		if embedding_vector is not None :
			embedding_matrix[index] = embedding_vector

	vocab_size = embedding_matrix.shape[0]

	return vocab_size, embedding_matrix
