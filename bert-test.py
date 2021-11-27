import torch
import tqdm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import TensorDataset

# Defining Model metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import BertForSequenceClassification
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
from transformers import BertTokenizer, AutoModelForSequenceClassification

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

savedUsed = True
saveFlag = False

# Reading  twitter  dataset
df = pd.read_csv("res/airlines.csv")
print(df[:4])
print(df.text.iloc[0])

# Counts of sentiments in the dataset
print(df.airline_sentiment.value_counts())

# Labels for targets
possible_label = df.airline_sentiment.unique()
dict_label = {}
for index, possible_label in enumerate(possible_label):
    dict_label[possible_label] = index
print(dict_label)

df["Label"] = df["airline_sentiment"].replace(dict_label)
print(df.head())

labels = ['negative','neutral','positive']
# Plot Labels
plt.rcParams['figure.figsize'] = (5, 5)
sns.countplot(df["Label"], hue=df["Label"], palette='dark')
plt.legend(loc='upper right')
plt.show()

# Plot Pie pices
labels = ['negative','neutral','positive']
#labels = [0, 1, 2]
sizes = df["Label"].value_counts()
colors = plt.cm.copper(np.linspace(0, 5, 9))
explode = [0.05, 0.05, 0.05]
cmap = plt.get_cmap('Spectral')
plt.pie(sizes, labels=labels, colors=colors, shadow=True, explode=explode)
plt.legend()
plt.show()




X_train, X_test, y_train, y_test = train_test_split(df.index.values,
                                                    df.Label.values,
                                                    test_size=0.15,
                                                    random_state=17,
                                                    stratify=df.Label.values)
df.loc[X_train, 'data_type'] = 'train'
df.loc[X_test, 'data_type'] = 'test'

print(df.head())
print(df.groupby(['airline_sentiment', 'Label', 'data_type']).count())


##
# Modeling
#
print('Modelling ...')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Encoding text by tokenizing using BERT Tokenizer
# In order to use BERT text embeddings as input to train text classification model,
# We need to tokenize our text reviews.
# Tokenization refers to dividing a sentence into individual words.
# To tokenize our text, we will be using the BERT tokenizer

encoder_train = tokenizer.batch_encode_plus(df[df["data_type"] == 'train'].text.values,
                                            add_special_tokens=True,
                                            return_attention_mask=True,
                                            padding=True,
                                            max_length=256,
                                            return_tensors='pt',
                                            truncation=True)

encoder_test = tokenizer.batch_encode_plus(df[df["data_type"] == 'test'].text.values,
                                           add_special_tokens=True,
                                           return_attention_mask=True,
                                           padding=True,
                                           max_length=256,
                                           return_tensors='pt',
                                           truncation=True)

input_ids_train = encoder_train['input_ids']
attention_masks_train = encoder_train["attention_mask"]
labels_train = torch.tensor(df[df['data_type'] == 'train'].Label.values)

input_ids_test = encoder_test['input_ids']
attention_masks_test = encoder_test["attention_mask"]
labels_test = torch.tensor(df[df['data_type'] == 'test'].Label.values)

data_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
data_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

print(len(data_train), len(data_test))


#
# We will use sequence classification model as we have
# to classify multi label text from the dataset.
#

print('model: BertForSequenceClassification.from_pretrained')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=len(dict_label),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# From torch we will use data loader,randomsampler to load data in an iterable format but
# extracting different subsamples from dataset.

dataloader_train = DataLoader(
    data_train,
    sampler=RandomSampler(data_train),
    batch_size=16

)

dataloader_test = DataLoader(
    data_test,
    sampler=RandomSampler(data_test),
    batch_size=32

)

optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

epochs = 1
#epochs = 10
print(f'Total epochs: {epochs}')
print(f'num_training_steps: {len(dataloader_train) * epochs}')

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(dataloader_train) * epochs
)


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels):
    label_dict_reverse = {v: k for k, v in dict_label.items()}
    print(f"Class:{label_dict_reverse}")

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class:{label}')
        print(f'Accuracy:{len(y_preds[y_preds == label])}/{len(y_true)}\n')


import random

seed_val = 171717
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
print(f'seed_val: {seed_val}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Loading:{device}")



# Validating data

def evaluate(dataloader_val):
    print(f'Evaluate model ....')
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in tqdm(dataloader_test):
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]
                  }
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


#Training  data
print(f'Training data ...')

for epoch in range(1, epochs + 1):
#for epoch in tqdm(range(1, epochs + 1)):
    model.train()

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch: {:1d}'.format(epoch), leave=True, disable=False)
    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]

        }
        outputs = model(**inputs)

        loss = outputs[0]
        #         logits = outputs[1]
        loss_train_total += loss.item()
        loss.backward()

        clip_arg  = 10
        norm_type = 2
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
        # To save the model after each epoch
        # torch.save(model.state_dict(),f'BERT_ft_epoch{epoch}.model')


    tqdm.write(f'\nEpoch  {epoch}')

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f'Training Loss: {loss_train_avg}')
    val_loss, predictions, true_vals = evaluate(dataloader_test)
    test_score = f1_score_func(predictions, true_vals)
    tqdm.write(f'Val Loss:{val_loss}\nTest Score: {test_score}')


if saveFlag == True:
    # Save the model
    print(f'Save BERT ft model ...')
    torch.save(model.state_dict(),f'BERT_ft.model')
else:
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                      num_labels = len(dict_label),
                                      output_attentions = False,
                                      output_hidden_states =  False)


model.to(device)

if savedUsed == True:
    # using saved model
    torch.load(model.state_dict(), f'BERT_ft.model')
    _,predictions,true_vals = evaluate(dataloader_test)
    accuracy_per_class(predictions,true_vals)
else:
    _, predictions, true_vals = evaluate(dataloader_test)
    accuracy_per_class(predictions, true_vals)



