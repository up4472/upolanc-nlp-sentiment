from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from typing import Any

from src.eval import evaluate_classification
from src.system import get_random_state

from collections import defaultdict

import numpy

MODELS = ['MNB', 'GNB', 'KNN', 'DT', 'RF', 'MV']

NAMES = {
	'MNB' : 'Naive Bayes (Multinomial)',
	'GNB' : 'Naive Bayes (Gaussian)',
	'KNN' : 'K-Nearest Neighbor',
	'DT' : 'Decision Tree',
	'RF' : 'Random Forest',
	'MV' : 'Majority Voting'
}

def create_classification (name : str) -> Any :
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
		], voting = 'soft')
	}[name]

def model_train (model : Any, xdata : numpy.ndarray, ydata : numpy.array) -> Any :
	model.fit(xdata, ydata)

	return model

def model_predict (model : Any, xdata : numpy.ndarray, ydata : numpy.array) ->  (numpy.ndarray, numpy.ndarray, numpy.ndarray):
	ypred = model.predict(xdata)
	yprob = model.predict_proba(xdata)

	return ydata, ypred, yprob

def models_kfold (xdata : numpy.ndarray, ydata : numpy.array, model_name : str, k_fold : int = 5) -> dict :
	# Create model
	model = create_classification(name = model_name)

	# Define KFold
	kf = StratifiedKFold(n_splits = k_fold, shuffle = True, random_state = get_random_state())

	# Empty list for results
	history = defaultdict(list)

	# KFold loop
	for train_index, test_index in kf.split(xdata, ydata) :
		xtrain, xtest = xdata[train_index], xdata[test_index]
		ytrain, ytest = ydata[train_index], ydata[test_index]

		# Train the model
		model = model_train(model = model, xdata = xtrain, ydata = ytrain)

		# Predict the model
		ytrue, ypred, yprob = model_predict(model = model, xdata = xtest, ydata = ytest)

		# Evaluate the model
		result = evaluate_classification(ytrue = ytrue, ypred = ypred, yprob = yprob)

		# Save the results
		history['test_accuracy'].append(result['accuracy_score'])
		history['test_precision'].append(result['precision'])
		history['test_recall'].append(result['recall'])
		history['test_f1_score'].append(result['f1_score'])
		history['test_brier_score'].append(result['brier_score'])

	return {
		'accuracy_score' : history['test_accuracy'],
		'precision' : history['test_precision'],
		'recall' : history['test_recall'],
		'f1_score' : history['test_f1_score'],
		'brier_score' : history['test_brier_score']
	}
