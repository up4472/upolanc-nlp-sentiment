from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import Module
from torch.utils.data import Dataset
from transformers import BertModel
from transformers import BertTokenizer

import numpy
import torch

class CustomDataset (Dataset) :

	def __init__ (self, text : numpy.ndarray, targets : numpy.ndarray, tokenizer : BertTokenizer, max_len : int) -> None :
		self.text = text
		self.targets = targets
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__ (self) -> int :
		return len(self.text)

	def __getitem__ (self, item) -> dict :
		text = str(self.text[item])
		target = self.targets[item]

		encoding = self.tokenizer.encode_plus(
			text = text,
			add_special_tokens = True,
			max_length = self.max_len,
			return_token_type_ids = False,
			pad_to_max_length = True,
			return_attention_mask = True,
			return_tensors = 'pt',
			truncation = True
		)

		return {
			'text' : text,
			'input_ids' : encoding['input_ids'].flatten(),
			'attention_mask' : encoding['attention_mask'].flatten(),
			'targets' : torch.tensor(target, dtype = torch.long)
		}

class BertForSentimentClassification (Module) :

	def __init__ (self, n_classes : int) -> None :
		super(BertForSentimentClassification, self).__init__()

		self.bert = BertModel.from_pretrained('bert-base-cased', return_dict = False)
		self.drop = Dropout(p = 0.3)
		self.out = Linear(self.bert.config.hidden_size, n_classes)

	def forward (self, input_ids, attention_mask) :
		_, pool = self.bert(
			input_ids = input_ids,
			attention_mask = attention_mask
		)

		return self.out(self.drop(pool))
