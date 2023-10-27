from utils import *
import torch
import evaluate
import datasets
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets.features.translation import Translation
from transformers import DataCollatorForSeq2Seq, MBart50TokenizerFast, MBartForConditionalGeneration


# Model Setting
model_name = "facebook/mbart-large-50-many-to-many-mmt"
# model_name = "facebook/nllb-200-distilled-600M"

# load Tokenizer
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="ko_KR")

# tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="ko_KR")
# tokenizer("Hello, this one sentence!")
# # {'input_ids': [250004, 35378, 4, 903, 1632, 149357, 38, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
# tokenizer(["Hello, this one sentence!", "This is another sentence."])
# # {'input_ids': [[250004, 35378, 4, 903, 1632, 149357, 38, 2], [250004, 3293, 83, 15700, 149357, 5, 2]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]}
# with tokenizer.as_target_tokenizer():
#     print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))
# # {'input_ids': [[250014, 35378, 4, 903, 1632, 149357, 38, 2], [250014, 3293, 83, 15700, 149357, 5, 2]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]}

model = MBartForConditionalGeneration.from_pretrained(model_name)
# model = MBartForConditionalGeneration.from_pretrained(model_name)
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
# Read Dataset from huggingface Dataset
data = datasets.load_dataset("riverkangg/aihub-ko-en-translation-socsci", 'train')
data = datasets.load_dataset("riverkangg/aihub-ko-en-translation-socsci", 'validation')
data = data.cast_column('translation', Translation(languages=['en', 'ko']))


# Make model input
def preprocess_function(examples, source_lang="en", target_lang="ko"):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    tokens = tokenizer(inputs, text_target=targets, max_length=128,
                       truncation=True, padding="max_length", return_tensors="pt")

    return tokens


print(preprocess_function(data['train'][:1]))
tokenized_datasets = data.map(preprocess_function, batched=True)
# train_tokenized = preprocess_function(data['train'])
# valid_tokenized = preprocess_function(data['validation'])

# Training Arguments 정의
batch_size = 1
source_lang = 'en'
target_lang = 'ko'
args = Seq2SeqTrainingArguments(
    f"mbart-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy="epoch",
    learning_rate=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    # fp16=True,
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
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
trainer.train()
trainer.save_model()
trainer.push_to_hub()
