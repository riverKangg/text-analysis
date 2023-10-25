from utils import *
import pandas as pd
import datasets
import torch
from torch.utils.data import Dataset

from torch.utils.data import TensorDataset

import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup, \
    TrainingArguments, Trainer

# Model Setting
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="ko_KR")
model = MBartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("mps")
model.to(device)

# Make dataset
data_dir = './data/aihub-ko-en'
col_list = ['data_set', 'domain', 'subdomain', 'ko', 'mt', 'en', 'source_language', 'target_language']
train = pd.read_csv(f'{data_dir}/1113_social_train_set_1210529.csv', usecols=col_list, nrows=300)
train.rename(columns={'en': 'source', 'ko': 'target'}, inplace=True)

# train_dataset = {}
# for i in range(len(train)):
#     tokens = tokenizer(train['source'].tolist()[i], text_target=train['target'].tolist()[i], max_length=1024,
#                        truncation=True, padding="max_length", return_tensors="pt")
#     train_dataset[i] = tokens

train_tokens = tokenizer(train['source'].tolist(), text_target=train['target'].tolist(), max_length=1024,
                         truncation=True, padding="max_length", return_tensors="pt")

valid = pd.read_csv(f'{data_dir}/1113_social_valid_set_151316.csv', usecols=col_list)
valid.rename(columns={'en': 'source', 'ko': 'target'}, inplace=True)
eval_tokens = tokenizer(train['source'].tolist(), text_target=train['target'].tolist(),
                        padding=True, truncation=True, max_length=512, return_tensors="pt")

# Make input tensor
train_dataset = datasets.DatasetDict({'input_ids': train_tokens['input_ids'],
                                      'attention_mask': train_tokens['attention_mask'],
                                      'labels': train_tokens['labels']})
eval_dataset = datasets.DatasetDict({'input_ids': eval_tokens['input_ids'].tolist(),
                                     'attention_mask': eval_tokens['attention_mask'].tolist(),
                                     'labels': eval_tokens['labels'].tolist()})

# Training Arguments 정의
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_dir="./logs",
    num_train_epochs=3,
)

# Trainer 생성
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    data_collator=None,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
)

trainer.train()
