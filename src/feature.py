from pandas import DataFrame

import numpy

def vader_features (dataset : DataFrame) -> (numpy.ndarray, numpy.ndarray) :
	yname = 'target'
	xname = ['vader_positive', 'vader_negative', 'vader_neutral', 'vader_compound']

	ydata = dataset[yname].to_numpy()
	xdata = dataset[xname].to_numpy()

	return xdata, numpy.ravel(ydata)

def vader_features_ext (dataset : DataFrame) -> (numpy.ndarray, numpy.ndarray) :
	yname = 'target'
	xname = ['vader_positive', 'vader_negative', 'vader_neutral', 'vader_compound', 'word_count', 'pos_count', 'neg_count']

	ydata = dataset[yname].to_numpy()
	xdata = dataset[xname].to_numpy()

	return xdata, numpy.ravel(ydata)
