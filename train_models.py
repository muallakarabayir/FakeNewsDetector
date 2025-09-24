# train_and_test.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1️⃣ CSV'den veriyi oku
df = pd.read_csv("data/processed/teyit_fake_news_clean.csv")

# 2️⃣ Etiketleri oluştur (0 = gerçek, 1 = sahte)
df['label'] = 0
# Örnek olarak bazı satırları sahte (1) yap
df.loc[[1, 4], 'label'] = 1

# 3️⃣ Özellik ve hedef değişken
X = df['clean_text']
y = df['label']

# 4️⃣ Eğitim ve test setine ayır (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ Metni vektörle
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6️⃣ Modeli oluştur ve eğit
model = LogisticRegression(max_iter=500, class_weight='balanced')
model.fit(X_train_vec, y_train)

# 7️⃣ Test setinde tahmin yap
y_pred = model.predict(X_test_vec)

# 8️⃣ Performans raporu
print("Test seti dağılımı:")
print(y_test.value_counts())
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# 9️⃣ Model ve vectorizer'ı kaydet
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel ve vectorizer 'models' klasörüne kaydedildi!")

#  🔟 İsteğe bağlı: test setinden örnek tahminler
for text, pred in zip(X_test, y_pred):
    print("\nHaber:", text[:80], "...")
    print("Tahmin:", "Sahte" if pred == 1 else "Gerçek")
