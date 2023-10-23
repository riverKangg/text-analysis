import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

INPUT_DIR = "../input/"
DATA_DIR = '../data/jigsaw/'
OUTPUT_DIR = '../output/jigsaw/'
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

MAX_SEQUENCE_LENGTH = 220
SEED = 1234
EPOCHS = 1
num_to_load = 100
valid_size = 10
TARGET_COL = 'target'

MODEL_NAME = "bert-base-uncased"
OUTPUT_MODEL_NAME = "bert_pytorch.bin"


# Converting the lines to BERT format
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


tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, cache_dir=None, do_lower_case=True)
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).sample(num_to_load + valid_size, random_state=SEED)
print('loaded %d records' % len(train_df))

sequences = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)

train_df = train_df.fillna(0)

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
y_columns = [TARGET_COL]

train_df = train_df.drop(['comment_text'], axis=1)
# convert target to 0,1
train_df[TARGET_COL] = (train_df[TARGET_COL] >= 0.5).astype(float)

X = sequences[:num_to_load]
y = train_df[y_columns].values[:num_to_load]
X_val = sequences[num_to_load:]
y_val = train_df[y_columns].values[num_to_load:]

test_df = train_df.tail(valid_size).copy()
train_df = train_df.head(num_to_load)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float))

# MODEL
OUTPUT_MODEL_FILE = "bert_pytorch.bin"
lr = 2e-5
batch_size = 32
accumulation_steps = 2
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=None, num_labels=len(y_columns))
device = torch.device("cpu")
model.to(device)

model.zero_grad()
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

train = train_dataset
num_train_optimization_steps = int(EPOCHS * len(train) / batch_size / accumulation_steps)
from transformers import get_linear_schedule_with_warmup

optimizer = AdamW(optimizer_grouped_parameters,
                  lr=lr)
# scheduler = WarmupLinearSchedule(optimizer, warmup_steps= num_train_optimization_steps,
#                                  t_total=num_train_optimization_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_optimization_steps,
                                            num_training_steps=num_train_optimization_steps)
model = model.train()

tq = tqdm(range(EPOCHS))
for epoch in tq:
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf = None
    tk0 = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    optimizer.zero_grad()

    for i, (x_batch, y_batch) in tk0:
        y_pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
        loss = F.binary_cross_entropy_with_logits(y_pred.logits, y_batch.to(device))
        if lossf:
            lossf = 0.98 * lossf + 0.02 * loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss=lossf)
        avg_loss += loss.item() / len(train_loader)

        # 정확도 계산 (CPU에서 계산)
        accuracy = torch.mean(((torch.sigmoid(y_pred[:, 0]) > 0.5) == (y_batch[:, 0] > 0.5).to(device)).to(torch.float))
        avg_accuracy += accuracy.item() / len(train_loader)

    tq.set_postfix(avg_loss=avg_loss, avg_accuracy=avg_accuracy)

torch.save(model.state_dict(), f'{OUTPUT_DIR}/{OUTPUT_MODEL_NAME}')

model_pretrained = BertForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=None, num_labels=len(y_columns))
model = BertForSequenceClassification(model_pretrained, num_labels=len(y_columns))
model.load_state_dict(torch.load(f'{OUTPUT_DIR}/{OUTPUT_MODEL_NAME}'))
model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()
valid_preds = np.zeros((len(X_val)))
valid = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.long))
valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)

tk0 = tqdm(valid_loader)
for i, (x_batch,) in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
    valid_preds[i * 32:(i + 1) * 32] = pred.logits.detach().cpu().squeeze().numpy()


# --
def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN] > 0.5
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive


def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup] > 0.5]
    return compute_auc((subgroup_examples[label] > 0.5), subgroup_examples[model_name])


def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[(df[subgroup] > 0.5) & (df[label] <= 0.5)]
    non_subgroup_positive_examples = df[(df[subgroup] <= 0.5) & (df[label] > 0.5)]
    examples = pd.concat([subgroup_negative_examples, non_subgroup_positive_examples])
    return compute_auc(examples[label] > 0.5, examples[model_name])


def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[(df[subgroup] > 0.5) & (df[label] > 0.5)]
    non_subgroup_negative_examples = df[(df[subgroup] <= 0.5) & (df[label] <= 0.5)]
    examples = pd.concat([subgroup_positive_examples, non_subgroup_negative_examples])
    return compute_auc(examples[label] > 0.5, examples[model_name])


def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup] > 0.5])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


MODEL_NAME = 'model1'
test_df[MODEL_NAME] = torch.sigmoid(torch.tensor(valid_preds)).numpy()
bias_metrics_df = compute_bias_metrics_for_model(test_df, identity_columns, MODEL_NAME, TARGET_COL)
bias_metrics_df
get_final_metric(bias_metrics_df, calculate_overall_auc(test_df, MODEL_NAME))