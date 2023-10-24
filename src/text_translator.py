from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TextTranslator:
    def __init__(self, model_name='facebook/nllb-200-distilled-600M'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # , use_auth_token=True
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name) # , use_auth_token=True

    def translate_text(self, text, target_language='kor_Hang', max_length=30):
        input_tokens = self.tokenizer(text, return_tensors="pt")
        translated_tokens = self.model.generate(
            **input_tokens, forced_bos_token_id=self.tokenizer.lang_code_to_id[target_language], max_length=max_length
        )
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text

if __name__ == "__main__":
    translator = TextTranslator()

    source_text = "Hello, how are you?"
    translated_text = translator.translate_text(source_text)

    print(f"번역 결과: {translated_text}")
