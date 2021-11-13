from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier

import numpy

def create_model (name : str) :
	return {
		'MNB' : MultinomialNB(),
		'GNB' : GaussianNB(),
		'DT' : DecisionTreeClassifier(),
		'KNN' : KNeighborsClassifier(),
		'RF' : RandomForestClassifier()
	}[name]

def train_model (xdata : numpy.ndarray, ydata : numpy.array, model_name : str, k_fold : int = 5) -> dict :
	# Create model
	model = create_model(name = model_name)

	# Define KFold
	kf = StratifiedKFold(n_splits = k_fold)

	# Empty list for results
	accuracy = list()
	brier = list()

	# KFold loop
	for train_index, test_index in kf.split(xdata, ydata) :
		xtrain, xtest = xdata[train_index], xdata[test_index]
		ytrain, ytest = ydata[train_index], ydata[test_index]

		# Train the model
		model.fit(xtrain, ytrain)

		# Predict the model
		ypred = model.predict(xtest)
		yprob = model.predict_proba(xtest)

		# Evaluate the model
		result = evaluate_classification(ytrue = ytest, ypred = ypred, yprob = yprob)

		# Save the results
		accuracy.append(result['accuracy_score'])
		brier.append(result['brier_score'])

	return {
		'accuracy_score' : accuracy,
		'brier_score' : brier
	}

def evaluate_classification (ytrue : numpy.ndarray, ypred : numpy.ndarray, yprob : numpy.ndarray = None) -> dict :
	_accuracy = accuracy_score(y_true = ytrue, y_pred = ypred)
	_confusion = confusion_matrix(y_true = ytrue, y_pred = ypred)
	_brier = numpy.nan

	if yprob is not None :
		lb = LabelBinarizer()

		_brier = numpy.sum(numpy.square(numpy.subtract(lb.fit_transform(ytrue), yprob))) / len(ytrue)

	return {
		'accuracy_score' : _accuracy,
		'confusion_matrix' : _confusion,
		'brier_score' : _brier
	}
