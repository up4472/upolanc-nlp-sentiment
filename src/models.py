from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from typing import Any

from src.eval import evaluate_classification

import numpy

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

def create_regression (name : str) -> Any :
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
		model = model_train(model = model, xdata = xtrain, ydata = ytrain)

		# Predict the model
		ytrue, ypred, yprob = model_predict(model = model, xdata = xtest, ydata = ytest)

		# Evaluate the model
		result = evaluate_classification(ytrue = ytrue, ypred = ypred, yprob = yprob)

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
