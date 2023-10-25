from utils import *
import pandas as pd
import datasets

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


valid = pd.read_csv(f'{data_dir}/1113_social_valid_set_151316.csv', usecols=col_list)
valid.rename(columns={'en': 'source', 'ko': 'target'}, inplace=True)
eval_tokens = tokenizer(train['source'].tolist(), text_target=train['target'].tolist(),
                         padding=True, truncation=True, max_length=512, return_tensors="pt")

# train_encodings = tokenizer(train["source"].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
# train_labels = tokenizer(train["target"].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")

import torch
from torch.utils.data import Dataset

from torch.utils.data import TensorDataset

# Make input tensor
train_tokens = tokenizer(train['source'].tolist(), text_target=train['target'].tolist(), max_length=1024,
                         truncation=True, padding="max_length", return_tensors="pt")
train_dataset = datasets.DatasetDict({'input_ids': train_tokens['input_ids'],
                                      'attention_mask': train_tokens['attention_mask'],
                                      'labels': train_tokens['labels']})

eval_dataset = datasets.DatasetDict({'input_ids': eval_tokens['input_ids'],
                                      'attention_mask': eval_tokens['attention_mask'],
                                      'labels': eval_tokens['labels']})


class CustomDataset(Dataset):
    def __init__(self, input_tensor, attention_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.attention_tensor = attention_tensor
        self.label_tensor = label_tensor

    def __len__(self):
        return len(self.input_tensor)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_tensor[idx],
            "attension_mask": self.attention_tensor[idx],
            "labels": self.label_tensor[idx]
        }


train_dataset = CustomDataset(train_input_ids, train_attention_mask, train_labels)

Dataset(train_tokens)
train_dataset = Dataset(train_input_ids, train_attention_mask, train_labels)


# TrainingArguments 정의
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
    eval_dataset=eval_dataset,
)

trainer.train()

MAX_SEQUENCE_LENGTH = 50
SEED = 1234
EPOCHS = 1
num_to_load = 100
valid_size = 10
TARGET_COL = 'target'
lr = 2e-5
batch_size = 32
accumulation_steps = 2
np.random.seed(SEED)
torch.manual_seed(SEED)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

num_train_optimization_steps = int(EPOCHS * len(inputs) / batch_size / accumulation_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_optimization_steps,
                                            num_training_steps=num_train_optimization_steps)

# Make input
# EPOCHS = 1
# batch_size = 32
# for i, (x_batch, y_batch) in enumerate(train_loader):
#     print(i)
# for epoch in range(EPOCHS):
#     train_loader = torch.utils.data.DataLoader(inputs, batch_size=batch_size, shuffle=True)
#     avg_loss = 0.
#     avg_accuracy = 0.
#     lossf = None
#     optimizer.zero_grad()
#
#     for i, (x_batch, y_batch) in enumerate(train_loader):
#         y_pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
#         loss = F.binary_cross_entropy_with_logits(y_pred.logits, y_batch.to(device))
#         if lossf:
#             lossf = 0.98 * lossf + 0.02 * loss.item()
#         else:
#             lossf = loss.item()
#         avg_loss += loss.item() / len(train_loader)
#
#         # 정확도 계산 (CPU에서 계산)
#         accuracy = torch.mean(((torch.sigmoid(y_pred[:, 0]) > 0.5) == (y_batch[:, 0] > 0.5).to(device)).to(torch.float))
#         avg_accuracy += accuracy.item() / len(train_loader)
