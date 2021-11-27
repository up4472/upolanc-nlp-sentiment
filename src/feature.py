from pandas import DataFrame
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from torch import LongTensor
from torch import Tensor

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

def bert_features (dataset : DataFrame) -> DataLoader :
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

	xdata = 'text'
	ydata = 'target'

	encoder = tokenizer.batch_encode_plus(dataset[xdata].values,
		add_special_tokens = True,
		return_attention_mask = True,
		padding = True,
		max_length = 256,
		return_tensors = 'pt',
		truncation = True
	)

	data = TensorDataset(encoder['input_ids'], encoder['attention_mask'], LongTensor(dataset[ydata].values))
	dataloader = DataLoader(
		dataset = data,
		sampler = RandomSampler(data),
		batch_size = 128
	)

	return dataloader
