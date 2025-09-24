# predict_news.py

import joblib

# Model ve vectorizer dosya yolları
MODEL_FILE = "models/fake_news_model.pkl"
VECTORIZER_FILE = "models/vectorizer.pkl"

# Kaydedilmiş model ve vectorizer'ı yükle
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# Tahmin yapmak istediğin haberler
news_samples = [
    "Bu haber gerçek mi, emin değilim.",
    "Ekrem İmamoğlu televizyona saldıranları gösteren video sahte."
]

for news in news_samples:
    # Metni vektörle
    news_vec = vectorizer.transform([news])
    # Tahmin
    pred = model.predict(news_vec)[0]
    # Sonucu yazdır
    print(f"Haber: {news}\nTahmin: {'Gerçek' if pred == 0 else 'Sahte'}\n")
