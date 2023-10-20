import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from utils import *

# 경로 및 상수 설정
INPUT_DIR = "./input/"
DATA_DIR = './data/jigsaw/'
OUTPUT_DIR = './output/jigsaw/'
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SEQUENCE_LENGTH = 220
SEED = 1234
EPOCHS = 1
# num_to_load = 100
# valid_size = 10
num_to_load = 10000
valid_size = 1000
TARGET_COL = 'target'
MODEL_NAME = "bert-base-uncased"
OUTPUT_MODEL_NAME = "bert_pytorch.bin"

# BERT 토크나이저 초기화
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, cache_dir=None, do_lower_case=True)

# 데이터 불러오기 및 전처리
dataset = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).sample(num_to_load + valid_size, random_state=SEED)
print(f'Loaded {len(dataset):,} records')
sequences = convert_lines(dataset["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)
dataset = dataset.fillna(0)

identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
                    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
y_columns = [TARGET_COL]

dataset = dataset.drop(['comment_text'], axis=1)
dataset[TARGET_COL] = (dataset[TARGET_COL] >= 0.5).astype(float)

# Split Train, Validation, Test Set
X = sequences[:num_to_load]
y = dataset[y_columns].values[:num_to_load]
X_val = sequences[num_to_load:]
y_val = dataset[y_columns].values[num_to_load:]

train_df = dataset.head(num_to_load)
test_df = dataset.tail(valid_size).copy()
print(f'Train Set: {len(X):,}\nValidation Set: {len(X_val):,}\nTest Set: {len(test_df):,}')

# 모델 및 학습 설정
OUTPUT_MODEL_FILE = "bert_pytorch.bin"
lr = 2e-5
batch_size = 32
accumulation_steps = 2
np.random.seed(SEED)
torch.manual_seed(SEED)

# 모델 초기화 및 GPU 설정
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=None, num_labels=len(y_columns))
dtype = torch.float
device = torch.device("mps")
model.to(device)

model.zero_grad()

for name, parmas in model.named_parameters():
    print(name)

param_list = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias',]
optim_params = [
    {'params': [p for n, p in param_list if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# TRAIN 모델 설정
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float))
num_train_optimization_steps = int(EPOCHS * len(train_dataset) / batch_size / accumulation_steps)
optimizer = AdamW(optim_params, lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_optimization_steps,
                                            num_training_steps=num_train_optimization_steps)
model = model.train()

# 학습 루프
print('START TRAINING')

for epoch in range(EPOCHS):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf = None
    optimizer.zero_grad()

    # tk0 = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    tq = tqdm(train_loader, total=len(train_loader), leave=True)
    for i, (x_batch, y_batch) in enumerate(tq):
        y_pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
        loss = F.binary_cross_entropy_with_logits(y_pred.logits, y_batch.to(device))
        if lossf:
            lossf = 0.98 * lossf + 0.02 * loss.item()
        else:
            lossf = loss.item()
        tq.set_postfix(loss=lossf)
        avg_loss += loss.item() / len(train_loader)
        accuracy = torch.mean(
            ((torch.sigmoid(y_pred.logits[:, 0]) > 0.5) == (y_batch[:, 0] > 0.5).to(device)).to(torch.float))
        avg_accuracy += accuracy.item() / len(train_loader)
    # tq.set_postfix(avg_loss=avg_loss, avg_accuracy=avg_accuracy)

print(f'AVG LOSS: {avg_loss} / AVE ACC: {avg_accuracy}')

# 모델 저장
torch.save(model.state_dict(), f'{OUTPUT_DIR}/{OUTPUT_MODEL_NAME}')

# 검증 데이터 예측
model_pretrained = BertForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=None, num_labels=len(y_columns))
model = BertForSequenceClassification(model_pretrained, num_labels=len(y_columns))
model.load_state_dict(torch.load(f'{OUTPUT_DIR}/{OUTPUT_MODEL_NAME}'))
model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()
valid_preds = np.zeros((len(X_val)))

# 검증 데이터 예측 루프
valid = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.long))
valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)
tk0 = tqdm(valid_loader)
for i, (x_batch,) in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
    valid_preds[i * 32:(i + 1) * 32] = pred.logits.detach().cpu().squeeze().numpy()

# Bias 메트릭을 계산
MODEL_NAME = 'model1'
test_df[MODEL_NAME] = torch.sigmoid(torch.tensor(valid_preds)).numpy()
bias_metrics_df = compute_bias_metrics_for_model(test_df, identity_columns, MODEL_NAME, TARGET_COL)
bias_metrics_df

# 최종 메트릭 계산
get_final_metric(bias_metrics_df, calculate_overall_auc(test_df, MODEL_NAME, TARGET_COL))
