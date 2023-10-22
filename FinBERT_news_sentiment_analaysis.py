import pandas as pd
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from finbert.finbert.finbert import predict
import warnings
import os
from datetime import datetime

# 경고 무시
warnings.filterwarnings('ignore')

# FinBERT 모델 및 토크나이저 불러오기
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=None, num_labels=3)
finbert_model.eval()

# 뉴스 데이터를 저장할 데이터프레임 생성
news_df = pd.DataFrame()

# 주식 정보를 가져올 티커 리스트
ticker_list = ['AAPL', 'O', 'TLT', 'SCHD', 'QQQ', 'VOO']

for ticker in ticker_list:
    # Yahoo Finance에서 뉴스 데이터 가져오기
    news_data = yf.Ticker(ticker).news
    news_titles = [article['title'] for article in news_data]

    for news_title in news_titles:
        # 뉴스 제목에 대한 감정 예측
        sentiment_result = predict(news_title, finbert_model)
        sentiment_result['ticker'] = ticker
        news_df = pd.concat([news_df, sentiment_result])

# 결과를 저장할 디렉토리 및 파일 이름 생성
output_dir = './output/news_sentiment_analysis/'
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime('%y%m%d')
output_file = f'{output_dir}/news_sentiment_analysis_{timestamp}.csv'

# 결과 데이터프레임을 CSV 파일로 저장
news_df.to_csv(output_file, index=False)