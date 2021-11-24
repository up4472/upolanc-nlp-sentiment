from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

import numpy

def create_classification (name : str) :
	return {
		'MNB' : MultinomialNB(),
		'GNB' : GaussianNB(),
		'KNN' : KNeighborsClassifier(),
		'DT' : DecisionTreeClassifier(),
		'RF' : RandomForestClassifier(),
		'MV' : VotingClassifier([
			('KNN', KNeighborsClassifier()),
			('DT', DecisionTreeClassifier()),
			('RF', RandomForestClassifier())
		])
	}[name]

def create_regression (name : str) :
	return {
		'KNN' : KNeighborsRegressor(),
		'DT' : DecisionTreeRegressor(),
		'RF' : RandomForestRegressor(),
		'MV' : VotingRegressor([
			('KNN', KNeighborsRegressor()),
			('DT', DecisionTreeRegressor()),
			('RF', RandomForestRegressor())
		])
	}[name]

def train_classification (xdata : numpy.ndarray, ydata : numpy.array, model_name : str, k_fold : int = 5) -> dict :
	# Create model
	model = create_classification(name = model_name)

	# Define KFold
	kf = StratifiedKFold(n_splits = k_fold)

	# Empty list for results
	accuracy = list()
	precision = list()
	recall = list()
	f1score = list()
	brier = list()

	# KFold loop
	for train_index, test_index in kf.split(xdata, ydata) :
		xtrain, xtest = xdata[train_index], xdata[test_index]
		ytrain, ytest = ydata[train_index], ydata[test_index]

		# Train the model
		model.fit(xtrain, ytrain)

		# Predict the model
		ypred = model.predict(xtest)

		if model_name in ['MV'] :
			yprob = None
		else :
			yprob = model.predict_proba(xtest)

		# Evaluate the model
		result = evaluate_classification(ytrue = ytest, ypred = ypred, yprob = yprob)

		# Save the results
		accuracy.append(result['accuracy_score'])
		precision.append(result['precision'])
		recall.append(result['recall'])
		f1score.append(result['f1_score'])
		brier.append(result['brier_score'])

	return {
		'accuracy_score' : accuracy,
		'precision' : precision,
		'recall' : recall,
		'f1_score' : f1score,
		'brier_score' : brier
	}

def evaluate_classification (ytrue : numpy.ndarray, ypred : numpy.ndarray, yprob : numpy.ndarray = None) -> dict :
	_accuracy = accuracy_score(y_true = ytrue, y_pred = ypred)
	_confusion = confusion_matrix(y_true = ytrue, y_pred = ypred)
	_metrics = precision_recall_fscore_support(y_true = ytrue, y_pred = ypred, average = 'weighted', zero_division = 0)
	_brier = numpy.nan

	_precision = _metrics[0]
	_recall = _metrics[1]
	_f1score = _metrics[2]

	if yprob is not None :
		lb = LabelBinarizer()

		_brier = numpy.sum(numpy.square(numpy.subtract(lb.fit_transform(ytrue), yprob))) / len(ytrue)

	return {
		'accuracy_score' : _accuracy,
		'precision' : _precision,
		'recall' : _recall,
		'f1_score' : _f1score,
		'brier_score' : _brier
	}
