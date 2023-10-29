import datasets
from transformers import MBart50TokenizerFast
from datasets.features.translation import Translation

path = "riverkangg/aihub-ko-en-translation-socsci"

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="ko_KR")


def tokenize_translation_data(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["ko"] for ex in examples["translation"]]
    tokens = tokenizer(inputs, text_target=targets, max_length=128,
                       truncation=True, padding=True, return_tensors="pt")
    return tokens


def load_translation_data(path=path):
    data = datasets.load_dataset(path)
    data = data.cast_column('translation', Translation(languages=['en', 'ko']))

    model_input = data.map(tokenize_translation_data, batched=True)

    return model_input


if __name__ == '__main__':
    path = "riverkangg/aihub-ko-en-translation-socsci"
    tokenized_datasets = load_translation_data(path=path)
    print(tokenized_datasets)
