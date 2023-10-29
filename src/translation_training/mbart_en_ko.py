import torch
import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq, MBart50TokenizerFast, MBartForConditionalGeneration

# Model Setting
model_name = "facebook/mbart-large-50-many-to-many-mmt"

# Load Tokenizer and Model
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="ko_KR")
model = MBartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("mps")
model.to(device)

# Load BLEU Metric
metric = evaluate.load('bleu')

# Load and Prepare Dataset
from utils.load_preprocessed_translation_data import load_translation_data

path = "riverkangg/aihub-ko-en-translation-socsci"
tokenized_datasets = load_translation_data(tokenizer, huggingface_path=path)

# Training Arguments
batch_size = 1
source_lang = 'en'
target_lang = 'ko'
training_args = Seq2SeqTrainingArguments(
    output_dir=f"mbart-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy="epoch",
    learning_rate=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
)

# Create Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Import custom utility functions from the 'utils' folder
from utils.compute_metrics import compute_metrics

# Create Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
)

# Train the Model
trainer.train()
trainer.save_model()
