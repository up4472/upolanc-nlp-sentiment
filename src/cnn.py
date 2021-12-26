from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History

from typing import Dict

import numpy

from src.eval import evaluate_classification

def create_cnn (input_dim : int, output_dim : int, input_size : int, output_size : int,
				embedding_matrix : numpy.ndarray = None) -> Sequential :

	model = Sequential()

	if embedding_matrix is None :
		model.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_size))
	else :
		model.add(Embedding(
			input_dim = embedding_matrix.shape[0],
			output_dim = embedding_matrix.shape[1],
			input_length = input_size,
			weights = [embedding_matrix])
		)

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

def cnn_train (model : Sequential, xdata : numpy.ndarray, ydata : numpy.ndarray, epochs : int, batch_size : int) -> History :
	history = model.fit(xdata, ydata,
		batch_size = batch_size,
		epochs = epochs,
		validation_split = 0.1
	)

	return history

def cnn_predict (model : Sequential, xdata : numpy.ndarray, ydata : numpy.ndarray, batch_size : int) -> Dict[str, float] :
	yprob = model.predict(xdata, batch_size = batch_size)

	ytrue = numpy.argmax(ydata, axis = 1).flatten()
	ypred = numpy.argmax(yprob, axis = 1).flatten()

	return evaluate_classification(ytrue = ytrue, ypred = ypred, yprob = yprob)
