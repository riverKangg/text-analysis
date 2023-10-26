from utils import *
import pandas as pd
import torch
import evaluate
import torch.nn.functional as F
from datasets import load_dataset, load_metric
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup, \
    DataCollatorForSeq2Seq

# Model Setting
model_name = "facebook/mbart-large-50-many-to-many-mmt"

# load Tokenizer
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="ko_KR")
# tokenizer("Hello, this one sentence!")
# # {'input_ids': [250004, 35378, 4, 903, 1632, 149357, 38, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
# tokenizer(["Hello, this one sentence!", "This is another sentence."])
# # {'input_ids': [[250004, 35378, 4, 903, 1632, 149357, 38, 2], [250004, 3293, 83, 15700, 149357, 5, 2]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]}
# with tokenizer.as_target_tokenizer():
#     print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))
# # {'input_ids': [[250014, 35378, 4, 903, 1632, 149357, 38, 2], [250014, 3293, 83, 15700, 149357, 5, 2]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]}

model = MBartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("mps")
model.to(device)

# load metric
metric = evaluate.load('bleu')  # load_metric("sacrebleu")
# check metric
# fake_preds = ["hello there", "안녕"]
# fake_labels = [["hello there"], ["안녕"]]
# metric.compute(predictions=fake_preds, references=fake_labels)
# {'bleu': 0.0, 'precisions': [1.0, 1.0, 0.0, 0.0], 'brevity_penalty': 1.0, 'length_ratio': 1.0,
# 'translation_length': 3, 'reference_length': 3}

# Read Data
data_dir = './data/aihub-ko-en'
# Make dataset
train = pd.read_csv(f'{data_dir}/1113_social_train_set_1210529.csv', usecols=['ko', 'en'])
train['translation'] = [{'en': x, 'ko': y} for x, y in zip(train['en'], train['ko'])]
trainset = Dataset.from_pandas(train[['translation']], split="train")

valid = pd.read_csv(f'{data_dir}/1113_social_valid_set_151316.csv', usecols=['ko', 'en'])
valid['translation'] = [{'en': x, 'ko': y} for x, y in zip(valid['en'], valid['ko'])]
validset = Dataset.from_pandas(valid[['translation']])


# Make model input
def preprocess_function(examples, source_lang="en", target_lang="ko"):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    tokens = tokenizer(inputs, text_target=targets, max_length=128,
                       truncation=True, padding="max_length", return_tensors="pt")

    model_inputs = Dataset.from_dict({'input_ids': tokens['input_ids'],
                                      'attention_mask': tokens['attention_mask'],
                                      'labels': tokens['labels']})

    return model_inputs


print(preprocess_function(trainset[:1]))

train_tokenized = preprocess_function(trainset[:100])
valid_tokenized = preprocess_function(validset[:10])

# Training Arguments 정의
batch_size = 16
source_lang = "en"
target_lang = "ko"
args = Seq2SeqTrainingArguments(
    f"MBART-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    # push_to_hub=True,
)

# Trainer 생성
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    compute_metrics=compute_metrics,
)

trainer.train()