import pandas as pd
import matplotlib.pyplot as plt

# Tahmin edilmiş haberleri oku
df = pd.read_csv("data/predicted_news.csv")

# label tahminlerini say
counts = df['prediction'].value_counts()

# Pasta grafiği
plt.figure(figsize=(6,6))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
plt.title("Haber Tahmin Dağılımı (Gerçek vs Sahte)")
plt.show()

# Alternatif: çubuk grafiği
plt.figure(figsize=(6,4))
plt.bar(counts.index, counts.values, color=['#66b3ff','#ff9999'])
plt.title("Haber Tahmin Dağılımı")
plt.ylabel("Haber Sayısı")
plt.show()
