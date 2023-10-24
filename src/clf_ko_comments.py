import os
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from datasets import load_dataset
from utils import *
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AdamW, get_linear_schedule_with_warmup)

# Setting Params
MAX_SEQUENCE_LENGTH = 220
SEED = 1234
num_to_load = 100
valid_size = 10
TARGET_COL = 'label'
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "bert-base-multilingual-cased"
OUTPUT_MODEL_NAME = "bert_multi_pytorch.bin"

# Setting Path
proj_dir = os.getcwd()
OUTPUT_DIR = f'{proj_dir}/model/clf_ko_comments/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setting logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.ERROR)

# Load Dataset
dataset_name = 'jeanlee/kmhas_korean_hate_speech'
dataset = load_dataset(dataset_name)
trainset = dataset["train"]

# Laad Tokenizer & Model
MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Convert text dataset to BERT input
def convert_lines(example, max_seq_length, tokenizer):
    max_seq_length -= 2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"]) + [0] * (
                max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)


sequences = convert_lines(trainset["text"], MAX_SEQUENCE_LENGTH, tokenizer)
labels = list(map(lambda x: int(8 not in x), trainset["label"]))
labels = np.array(labels).reshape(-1, 1)

# Split Train, Validation, Test set
train_size = round(len(sequences) * 0.6)
X = sequences[:train_size]
y = labels[:train_size]
X_val = sequences[train_size:]
y_val = labels[train_size:]

# val_size = round(len(sequences) * 0.8)
# X_val = sequences[train_size:val_size]
# y_val = labels[train_size:val_size]
# X_test = sequences[val_size:]
# y_test = labels[val_size:]

# Convert dataset to torch
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float))

# Load Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=None, num_labels=2)
device = torch.device("mps")
model.to(device)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

EPOCHS = 1
lr = 2e-5
batch_size = 10
accumulation_steps = 2
num_train_optimization_steps = int(EPOCHS * len(X) / batch_size / accumulation_steps)

optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_optimization_steps,
                                            num_training_steps=num_train_optimization_steps)

# Change to train mode
model = model.train()
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
# Training
tq = tqdm(range(EPOCHS))
for epoch in tq:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf = None
    optimizer.zero_grad()

    for i, (x_batch, y_batch) in enumerate(train_loader):
        y_pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None).logits
        y_pred_view = y_pred[:, 1].view(-1, 1)
        loss = F.binary_cross_entropy_with_logits(y_pred_view, y_batch.to(device))
        if lossf:
            lossf = 0.98 * lossf + 0.02 * loss.item()
        else:
            lossf = loss.item()
        avg_loss += loss.item() / len(train_loader)

        accuracy = torch.mean(((torch.sigmoid(y_pred[:, 0]) > 0.5) == (y_batch[:, 0] > 0.5).to(device)).to(torch.float))
        avg_accuracy += accuracy.item() / len(train_loader)

    tq.set_postfix(avg_loss=avg_loss, avg_accuracy=avg_accuracy)

# Save Model
torch.save(model.state_dict(), f'{OUTPUT_DIR}/{OUTPUT_MODEL_NAME}')
print('Model Save Done!')

# Evaluate Model
model.eval()

valid_preds = np.zeros((len(X_val)))
valid = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.long))
valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)

for i, (x_batch,) in enumerate(valid_loader):
    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
    valid_preds[i * 32:(i + 1) * 32] = pred.logits.detach().cpu().squeeze().numpy()

valid_preds_sigmoid = torch.sigmoid(torch.tensor(valid_preds)).numpy()
compute_auc(y_val, valid_preds_sigmoid)
