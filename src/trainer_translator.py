from utils import *
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch.nn.functional as F
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup

# Model Setting
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="ko_KR")
model = MBartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("mps")
model.to(device)

# Read Data
data_dir = './data/aihub-ko-en'
col_list = ['data_set', 'domain', 'subdomain', 'ko', 'mt', 'en', 'source_language', 'target_language']
train = pd.read_csv(f'{data_dir}/1113_social_train_set_1210529.csv', usecols=col_list, nrows=300)
valid = pd.read_csv(f'{data_dir}/1113_social_valid_set_151316.csv', usecols=col_list)

# Preprocessing Data
class CustomDataset(Dataset):
    def __init__(self, input_tensors, attention_tensors, label_tensors):
        self.input_tensors = input_tensors
        self.attention_tensors = attention_tensors
        self.label_tensors = label_tensors

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, index):
        input_data = self.input_tensors[index]
        attention_data = self.attention_tensors[index]
        label_data = self.label_tensors[index]
        return input_data, attention_data, label_data


train_tokens = tokenizer(train['ko'].tolist(), text_target=train['en'].tolist(), max_length=128,
                         truncation=True, padding="max_length", return_tensors="pt")
train_dataset = CustomDataset(train_tokens['input_ids'], train_tokens['attention_mask'], train_tokens['labels'])

eval_tokens = tokenizer(train['ko'].tolist(), text_target=train['en'].tolist(),
                        padding=True, truncation=True, max_length=128, return_tensors="pt")
eval_dataset = CustomDataset(eval_tokens['input_ids'], eval_tokens['attention_mask'], eval_tokens['labels'])

# Training Arguments 정의
args = Seq2SeqTrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_dir="./logs",
    num_train_epochs=3,
)

batch_size = 64
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# data_loader를 이용하여 데이터를 반복적으로 가져올 수 있음
for batch in data_loader:
    input_data = batch["input"]
    label_data = batch["label"]
# Trainer 생성
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    data_collator=None,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
)

trainer.train()
