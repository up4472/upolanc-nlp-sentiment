from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import AdamW
from pandas import DataFrame
from typing import Any

from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
from matplotlib import pyplot
from tqdm import tqdm

from src.classes import BertForSentimentClassification
from src.classes import CustomDataset
from src.eval import evaluate_classification
from src.system import get_random_state
from src.system import get_device

import torch
import numpy

BATCH_SIZE = 128
MAX_LEN = 100

def create_bert (name : str, num_labels : int) -> Any :
	return {
		'sentiment' :
			BertForSentimentClassification(n_classes = num_labels),
	}[name.lower()]

def create_dataloader (dataset : DataFrame, tokenizer, max_len : int, batch_size : int) -> DataLoader :
	dataset = CustomDataset(
		text = dataset.text.to_numpy(),
		targets = dataset.target.to_numpy(),
		tokenizer = tokenizer,
		max_len = max_len
	)

	return DataLoader(
		dataset = dataset,
		batch_size = batch_size,
		num_workers = 4
	)

def bert_train (model : Any, dataloader : DataLoader, models : dict) :
	optimizer = models['optimizer']
	scheduler = models['scheduler']
	device = models['device']
	loss_fn = models['loss_fn']

	# Set model to training mode
	model.train()

	losses = []
	ytrue = []
	ypred = []
	yprob = []

	for batch in tqdm(dataloader) :
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		targets = batch['targets'].to(device)

		outputs = model(
			input_ids = input_ids,
			attention_mask = attention_mask
		)

		_, pred = torch.max(outputs, dim = 1)
		prob = torch.softmax(outputs, dim = 1)

		lossval = loss_fn(outputs, targets)

		losses.append(lossval.item())

		lossval.backward()
		clip_grad_norm_(model.parameters(), max_norm = 1.0)

		optimizer.step()
		scheduler.step()
		optimizer.zero_grad()

		ytrue.append(targets.numpy())
		ypred.append(pred.numpy())
		yprob.append(prob.detach().numpy())

	ytrue = numpy.concatenate(ytrue, axis = 0)
	ypred = numpy.concatenate(ypred, axis = 0)
	yprob = numpy.concatenate(yprob, axis = 0)

	loss = numpy.mean(losses)

	return model, ytrue, ypred, yprob, loss

def bert_predict (model : Any, dataloader : DataLoader, models : dict) :
	device = models['device']
	loss_fn = models['loss_fn']

	# Set model to evaluation mode
	model.eval()

	losses = []
	ytrue = []
	ypred = []
	yprob = []

	with torch.no_grad() :
		for batch in tqdm(dataloader) :
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			targets = batch['targets'].to(device)

			outputs = model(
				input_ids = input_ids,
				attention_mask = attention_mask
			)

			_, pred = torch.max(outputs, dim = 1)
			prob = torch.softmax(outputs, dim = 1)

			lossval = loss_fn(outputs, targets)

			losses.append(lossval.item())

			ytrue.append(targets.numpy())
			ypred.append(pred.numpy())
			yprob.append(prob.numpy())

	ytrue = numpy.concatenate(ytrue, axis = 0)
	ypred = numpy.concatenate(ypred, axis = 0)
	yprob = numpy.concatenate(yprob, axis = 0)

	loss = numpy.mean(losses)

	return model, ytrue, ypred, yprob, loss

def bert_defsplit (dataset : DataFrame, name : str = 'sentiment', save_model : bool = True, epochs : int = 1) -> dict :
	# Create device
	device = get_device()

	# Lock random
	numpy.random.seed(get_random_state())
	torch.manual_seed(get_random_state())

	# Split training and testing set
	dataset_train, dataset_test = train_test_split(dataset,
		test_size = 0.33,
		random_state = get_random_state(),
		stratify = dataset.target.values
	)

	# Create tokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased', truncation = True)

	# Create train and test data loader
	dataloader_train = create_dataloader(
		dataset = dataset_train,
		tokenizer = tokenizer,
		max_len = MAX_LEN,
		batch_size = BATCH_SIZE
	)

	dataloader_test = create_dataloader(
		dataset = dataset_test,
		tokenizer = tokenizer,
		max_len = MAX_LEN,
		batch_size = BATCH_SIZE
	)

	# Create model
	model = create_bert(name = name, num_labels = len(dataset['target'].unique()))
	model = model.to(device)

	# Create optimizer function
	optimizer = AdamW(model.parameters(), lr = 1e-5, correct_bias = False)

	# Create scheduler
	scheduler = get_linear_schedule_with_warmup(
		optimizer = optimizer,
		num_warmup_steps = 0,
		num_training_steps = epochs * len(dataloader_train)
	)

	# Create loss function
	loss_fn = CrossEntropyLoss().to(device)

	# Create lookup table
	models = {
		'device' : device,
		'optimizer' : optimizer,
		'scheduler' : scheduler,
		'loss_fn' : loss_fn.float()
	}

	# Empty list for results
	history = defaultdict(list)
	best_acc = 0

	for epoch in range(epochs) :
		# Train the model
		model, ytrue, ypred, yprob, loss = bert_train(model = model, dataloader = dataloader_train, models = models)

		# Save the results
		result = evaluate_classification(ytrue = ytrue, ypred = ypred, yprob = yprob)

		history['train_loss'].append(loss)

		history['train_accuracy'].append(result['accuracy_score'])
		history['train_precision'].append(result['precision'])
		history['train_recall'].append(result['recall'])
		history['train_f1_score'].append(result['f1_score'])
		history['train_brier_score'].append(result['brier_score'])

		# Predict the model
		model, ytrue, ypred, yprob, loss = bert_predict(model = model, dataloader = dataloader_test, models = models)

		# Evaluate the model
		result = evaluate_classification(ytrue = ytrue, ypred = ypred, yprob = yprob)

		# Save the results
		history['test_loss'].append(loss)

		history['test_accuracy'].append(result['accuracy_score'])
		history['test_precision'].append(result['precision'])
		history['test_recall'].append(result['recall'])
		history['test_f1_score'].append(result['f1_score'])
		history['test_brier_score'].append(result['brier_score'])

		if save_model and result['accuracy_score'] > best_acc :
			torch.save(model.state_dict(), f'res\\bert_{name}_model.dat')
			best_acc = result['accuracy_score']

	xdata = numpy.arange(1, 1 + epochs)

	pyplot.figure()
	pyplot.plot(xdata, history['train_accuracy'], label = 'Train')
	pyplot.plot(xdata, history['test_accuracy'], label = 'Test')

	pyplot.title('Accuracy History')
	pyplot.ylabel('Accuracy')
	pyplot.xlabel('Epoch')
	pyplot.legend()
	pyplot.ylim([0, 1])
	pyplot.savefig(f'out\\bert_{name}_accuracy.png')

	pyplot.figure()
	pyplot.plot(xdata, history['train_loss'], label = 'Train')
	pyplot.plot(xdata, history['test_loss'], label = 'Test')

	pyplot.title('Loss History')
	pyplot.ylabel('Loss')
	pyplot.xlabel('Epoch')
	pyplot.legend()
	pyplot.ylim([0, 1])
	pyplot.savefig(f'out\\bert_{name}_loss.png')

	return {
		'accuracy_score' : history['test_accuracy'],
		'precision' : history['test_precision'],
		'recall' : history['test_recall'],
		'f1_score' : history['test_f1_score'],
		'brier_score' : history['test_brier_score'],
		'loss' : history['test_loss']
	}
