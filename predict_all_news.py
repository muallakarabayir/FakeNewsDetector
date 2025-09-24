# predict_all_news.py

import pandas as pd
import joblib

# Model ve vectorizer dosya yolları
MODEL_FILE = "models/fake_news_model.pkl"
VECTORIZER_FILE = "models/vectorizer.pkl"

# Model ve vectorizer'ı yükle
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# CSV'deki tüm haberleri oku
df = pd.read_csv("data/processed/teyit_fake_news_clean.csv")

# Tahmin yapılacak metinler
texts = df['clean_text'].tolist()

# Tahmin et
preds = model.predict(vectorizer.transform(texts))

# Tahminleri dataframe'e ekle
df['prediction'] = ['Gerçek' if p == 0 else 'Sahte' for p in preds]

# Sonucu göster
print(df[['title', 'prediction']])

# İstersen sonucu CSV olarak kaydet
df.to_csv("data/predicted_news.csv", index=False)
print("\nTahminler 'data/predicted_news.csv' dosyasına kaydedildi!")
