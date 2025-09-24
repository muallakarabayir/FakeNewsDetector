import pandas as pd
import re

# CSV oku
df = pd.read_csv("data/processed/teyit_fake_news.csv")

# Küçük harf
def lower_text(text):
    return text.lower()

# Noktalama kaldır
def remove_punctuation(text):
    return re.sub(r'[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ\s]', '', text)

# Basit Türkçe stopword
stopwords = ['ve', 'bir', 'bu', 'da', 'için', 'ile', 'o', 'mi', 'mü', 'ne', 'ki']

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stopwords])

def clean_text(text):
    text = lower_text(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text

df['text'] = df['title'] + " " + df['summary']
df['clean_text'] = df['text'].apply(clean_text)

df.to_csv("data/processed/teyit_fake_news_clean.csv", index=False, encoding="utf-8-sig")
print("Tamam, temizlenmiş veri kaydedildi!")
