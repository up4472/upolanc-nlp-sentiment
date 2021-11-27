from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import AdamW
from pandas import DataFrame
from typing import Any

from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.eval import evaluate_classification
from src.feature import bert_features

import torch
import numpy

def create_bert (num_labels : int) -> BertForSequenceClassification :
	return BertForSequenceClassification.from_pretrained('bert-base-uncased',
		num_labels = num_labels,
		output_attentions = False,
		output_hidden_states = False
	)

def bert_train (model : Any, dataloader : DataLoader, models : dict) -> Any :
	optimizer = models['optimizer']
	scheduler = models['scheduler']
	device = models['device']

	# Set model to training mode
	model.train()

	loss_total = 0

	for batch in tqdm(dataloader) :
		model.zero_grad()

		batch = tuple(item.to(device) for item in batch)

		inputs = {
			'input_ids' : batch[0],
			'attention_mask' : batch[1],
			'labels' : batch[2]
		}

		outputs = model(**inputs)

		loss = outputs[0]

		loss_total = loss_total + loss.item()

		clip_grad_norm_(model.parameters(), 1.0)

		optimizer.step()
		scheduler.step()

	return model

def bert_predict (model : Any, dataloader : DataLoader, models : dict) -> (Any, numpy.ndarray, numpy.ndarray) :
	device = models['device']

	# Set model to evaluation mode
	model.eval()

	loss_total = 0
	yprob = []
	ytrue = []

	for batch in tqdm(dataloader) :
		batch = tuple(item.to(device) for item in batch)

		inputs = {
			'input_ids' : batch[0],
			'attention_mask' : batch[1],
			'labels' : batch[2]
		}

		with torch.no_grad() :
			outputs = model(**inputs)

		loss = outputs[0]
		logits = outputs[1]

		loss_total = loss_total + loss.item()

		yprob.append(logits.detach().cpu().numpy())
		ytrue.append(inputs['labels'].cpu().numpy())

	ytrue = numpy.concatenate(ytrue, axis = 0)
	yprob = numpy.concatenate(yprob, axis = 0)
	ypred = numpy.argmax(yprob, axis = 1).flatten()

	return model, ytrue, ypred, yprob

def bert_defsplit (dataset : DataFrame, save_model : bool = True, epochs : int = 1) -> dict :
	# Create device
	device = torch.device('cpu')

	# Split training and testing set
	xtrain, xtest, ytrain, ytest = train_test_split(dataset.index.values, dataset.target.values,
		train_size = 0.67,
		random_state = None,
		stratify = dataset.target.values
	)

	# Add data type marker
	dataset.loc[xtrain, 'type'] = 'train'
	dataset.loc[xtest, 'type'] = 'test'

	# Create train and test data loader
	dataloader_train = bert_features(dataset = dataset[dataset['type'] == 'train'])
	dataloader_test = bert_features(dataset = dataset[dataset['type'] == 'test'])

	# Create model
	model = create_bert(num_labels = len(dataset['target'].unique()))

	# Create optimizer function
	optimizer = AdamW(model.parameters(), lr = 1e-5, eps = 1e-8)

	# Create scheduler
	scheduler = get_linear_schedule_with_warmup(
		optimizer = optimizer,
		num_warmup_steps = 0,
		num_training_steps = epochs * len(dataloader_train)
	)

	# Create lookup table
	models = {
		'device' : device,
		'optimizer' : optimizer,
		'scheduler' : scheduler
	}

	# Transform model for device type
	model = model.to(device = device)

	# Empty list for results
	accuracy_class = numpy.zeros(shape = (len(dataset['target'].unique()),), dtype = list)
	loss_avg_train = list()
	loss_avg_test = list()
	accuracy = list()
	precision = list()
	recall = list()
	f1score = list()
	brier = list()

	for label in numpy.unique(dataset['target']) :
		accuracy_class[label] = list()

	def accuracy_per_class (y, x) :
		acc_result = list()

		for item in sorted(numpy.unique(y)) :
			y_pred = x[y == item]
			y_true = y[y == item]

			acc_result.append(len(y_pred[y_pred == item]) / len(y_true))

		return acc_result

	for epoch in range(epochs) :
		# Train the model
		model = bert_train(model = model, dataloader = dataloader_train, models = models)

		# Predict the model
		model, ytrue, ypred, yprob = bert_predict(model = model, dataloader = dataloader_test, models = models)

		# Print out accuracy per class
		for index, acc in enumerate(accuracy_per_class(y = ytrue, x = ypred)) :
			accuracy_class[index].append(acc)

		# Evaluate the model
		result = evaluate_classification(ytrue = ytrue, ypred = ypred, yprob = yprob)

		# Save the results
		accuracy.append(result['accuracy_score'])
		precision.append(result['precision'])
		recall.append(result['recall'])
		f1score.append(result['f1_score'])
		brier.append(result['brier_score'])

	if save_model :
		torch.save(model.state_dict(), 'out\\bert_model.dat')

	return {
		'accuracy_per_class' : accuracy_class,
		'loss_train_avg' : loss_avg_train,
		'loss_test_avg' : loss_avg_test,
		'accuracy_score' : accuracy,
		'precision' : precision,
		'recall' : recall,
		'f1_score' : f1score,
		'brier_score' : brier
	}
