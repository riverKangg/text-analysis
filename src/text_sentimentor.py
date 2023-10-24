import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextSentimentor:
    def __init__(self, model_name="ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=None, num_labels=3)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - x.max())
        return e_x / e_x.sum()

    def sentiment_text(self, text, return_logit=False):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # logits = outputs.logits
        logits = self.softmax(outputs.logits)
        predicted_class = torch.argmax(logits, dim=1).item()

        if predicted_class == 0:
            sentiment = "Positive"
        elif predicted_class == 1:
            sentiment = "Negative"
        elif predicted_class == 2:
            sentiment = "Neutral"
        else:
            sentiment = "Unknown"

        if return_logit:
            return logits, sentiment
        else:
            return sentiment

if __name__ == "__main__":
    ts = TextSentimentor()

    text_list = ["Shares in the spin-off of South African e-commerce group Naspers surged more than 25% in the first minutes of their market debut in Amsterdam on Wednesday.",
                 "Bob van Dijk, CEO of Naspers and Prosus Group poses at Amsterdam's stock exchange, as Prosus begins trading on the Euronext stock exchange in Amsterdam, Netherlands, September 11, 2019."]

    for source_text in text_list:
        logit, sentiment_text = ts.sentiment_text(source_text, return_logit=True)
        print(f"{source_text}\n: {sentiment_text}, {logit}")