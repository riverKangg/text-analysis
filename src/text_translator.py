from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import corpus_bleu
import nltk

trans_models = ['facebook/nllb-200-distilled-600M',
          'facebook/mbart-large-50-many-to-many-mmt']
class TextTranslator:
    def __init__(self, model_name='facebook/nllb-200-distilled-600M'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def translate_text(self, text, target_language='kor_Hang', max_length=30):
        input_tokens = self.tokenizer(text, return_tensors="pt")
        translated_tokens = self.model.generate(
            **input_tokens, forced_bos_token_id=self.tokenizer.lang_code_to_id[target_language], max_length=max_length
        )
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text

    def evaluate_blue_text(self):
        nltk.download('punkt')

        references = [['참조 번역 1', '참조 번역 2']]
        hypothesis = ['모델 출력 번역']

        bleu_score = corpus_bleu(references, hypothesis)

        print(f'한국어 BLEU 스코어: {bleu_score * 100:.2f}')


if __name__ == "__main__":
    translator = TextTranslator()

    # source_text = "Hello, how are you?"
    source_text = """Well-known large-cap stocks are often the ones that get all the media attention, but sometimes it's the smaller companies are performing the best while the big names are struggling.

            To some degree, that's what's happening in the real estate investment trust (REIT) sector in recent weeks, as big names like Realty Income Corp. (NYSE:O), W.P. Carey Inc. (NYSE:WPC), SL Green Realty Corp. (NYSE:SLG), Prologis Inc. (NYSE:PLD) and others have been slashed in price or are just treading water. Meanwhile, some smaller REITs are sneaking up in price without anyone taking notice.

            Take a look at three small-cap REITs that have been performing well that investors may want to put on their radar screens.

            Strawberry Fields REIT Inc. (NYSEAMERICAN: STRW) is a self-managed and self-administered healthcare REIT that owns and operates 83 triple-net skilled nursing facilities, assisted living and other post-acute healthcare properties in Arkansas, Illinois, Indiana, Kentucky, Michigan, Ohio and a few other states. Strawberry Fields REIT was founded in 2004. After trading over the counter for a long time, it began trading on the New York Stock Exchange (NYSE) in February. Its market cap is $340.29 million.

            Strawberry Fields pays a quarterly dividend of $0.11 per share. The annual dividend of $0.44 yields 6.29%. Its payout ratio is a manageable 67.3%.

            On Aug. 15, Strawberry Fields REIT announced its second-quarter operating results. Funds from operations (FFO) of $12.7 million was up from $12.6 million in the second quarter of 2022, and rental income rose to $24.3 million in the second quarter of 2023 from $21.8 million in the second quarter of 2022.

            There has been no recent news to propel this stock higher, but over the past five trading days, Strawberry Fields REIT has led all REITs with a 14.31% return. Healthcare REITs in general have been outperforming the other REIT subsectors in recent weeks.
                """
    translated_text = translator.translate_text(source_text)

    print(f"번역 결과: {translated_text}")
