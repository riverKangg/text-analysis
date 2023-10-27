import datasets
from datasets.features.translation import Translation

huggingface_path = "riverkangg/aihub-ko-en-translation-socsci"
data = datasets.load_dataset(huggingface_path)


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["ko"] for ex in examples["translation"]]
    tokens = tokenizer(inputs, text_target=targets, max_length=128,
                       truncation=True, padding=True, return_tensors="pt")
    return tokens


def LoadPreprocessedTranslationData(tokenizer, huggingface_path="riverkangg/aihub-ko-en-translation-socsci"):
    data = datasets.load_dataset(huggingface_path)
    data = data.cast_column('translation', Translation(languages=['en', 'ko']))

    model_input = data.map(preprocess_function, batched=True)

    return model_input


if __name__ == '__main__':
    from transformers import MBart50TokenizerFast

    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="ko_KR")

    path = "riverkangg/aihub-ko-en-translation-socsci"
    tokenized_datasets = LoadPreprocessedTranslationData(tokenizer, huggingface_path=path)

    print(tokenized_datasets)
