from nltk.translate.bleu_score import corpus_bleu
import nltk
from utils import *
import pandas as pd
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration


class TextTranslator:
    def __init__(self, model_name="facebook/mbart-large-50-many-to-many-mmt"):
        self.base_model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        self.tokenizer.src_lang = "en_XX"

    def translate_text(self, text, tuned_model=None):
        if tuned_model:
            model = tuned_model
        else:
            model = self.base_model
        encoded = self.tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["ko_KR"]
        )
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translated_text

    def prepare_dataset(self):
        data_dir = './data/aihub-ko-en/'
        col_list = ['data_set', 'domain', 'subdomain', 'ko', 'mt', 'en', 'source_language', 'target_language']
        train = pd.read_csv(f'{data_dir}/1113_social_train_set_1210529.csv', usecols=col_list)
        test = pd.read_csv(f'{data_dir}/1113_social_valid_set_1210529.csv', usecols=col_list)

        train_token = self.tokenizer(train, return_tensors="pt", padding=True, truncation=True)
        test_token = self.tokenizer(test, return_tensors="pt", padding=True, truncation=True)

        train_dataset = CustomDataset(train_token)
        test_dataset = CustomDataset(test_token)

        return train_dataset, test_dataset

    def training(self):
        train_dataset, test_dataset = self.prepare_dataset()

        # 학습 설정 및 Trainer 설정
        training_args = Seq2SeqTrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
            num_train_epochs=3,
            logging_dir="./logs",
            logging_steps=100,
        )

        trainer = Seq2SeqTrainer(
            model=self.base_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        trainer.train()

        trainer.save_model("./model/fine_tuned_bart")

        results = trainer.predict(test_dataset)
        print(results.predictions)


if __name__ == "__main__":
    translator = TextTranslator()

    # source_text = "Hello, how are you?"
    source_text = """Well-known large-cap stocks are often the ones that get all the media attention, but sometimes it's the smaller companies are performing the best while the big names are struggling.\
            To some degree, that's what's happening in the real estate investment trust (REIT) sector in recent weeks, as big names like Realty Income Corp. (NYSE:O), W.P. Carey Inc. (NYSE:WPC), SL Green Realty Corp. (NYSE:SLG), Prologis Inc. (NYSE:PLD) and others have been slashed in price or are just treading water. Meanwhile, some smaller REITs are sneaking up in price without anyone taking notice.\
            Take a look at three small-cap REITs that have been performing well that investors may want to put on their radar screens.\
            Strawberry Fields REIT Inc. (NYSEAMERICAN: STRW) is a self-managed and self-administered healthcare REIT that owns and operates 83 triple-net skilled nursing facilities, assisted living and other post-acute healthcare properties in Arkansas, Illinois, Indiana, Kentucky, Michigan, Ohio and a few other states. Strawberry Fields REIT was founded in 2004. After trading over the counter for a long time, it began trading on the New York Stock Exchange (NYSE) in February. Its market cap is $340.29 million.\
            Strawberry Fields pays a quarterly dividend of $0.11 per share. The annual dividend of $0.44 yields 6.29%. Its payout ratio is a manageable 67.3%.\
            On Aug. 15, Strawberry Fields REIT announced its second-quarter operating results. Funds from operations (FFO) of $12.7 million was up from $12.6 million in the second quarter of 2022, and rental income rose to $24.3 million in the second quarter of 2023 from $21.8 million in the second quarter of 2022.\
            There has been no recent news to propel this stock higher, but over the past five trading days, Strawberry Fields REIT has led all REITs with a 14.31% return. Healthcare REITs in general have been outperforming the other REIT subsectors in recent weeks.\
                """
    translated_text = translator.translate_text(source_text)

    print(f"번역 결과: {translated_text}")
