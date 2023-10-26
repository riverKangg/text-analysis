import pandas as pd
from datasets import load_dataset

# Read Data
data_dir = './data/aihub-ko-en'

train = pd.read_csv(f'{data_dir}/1113_social_train_set_1210529.csv', usecols=['ko', 'en'])
train['translation'] = [{'en': x, 'ko': y} for x, y in zip(train['en'], train['ko'])]
train[['translation']].to_csv('./train.csv', index=False)

valid = pd.read_csv(f'{data_dir}/1113_social_valid_set_151316.csv', usecols=['ko', 'en'])
valid['translation'] = [{'en': x, 'ko': y} for x, y in zip(valid['en'], valid['ko'])]
valid[['translation']].to_csv('./validation.csv', index=False)
## --- upload files

# load Dataset from huggingface dataset
dataset = load_dataset("riverkangg/aihub-ko-en-translation-socsci")
