# backend/utils/fetch_teyit_data.py

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd
import time

# ChromeDriver'ı otomatik yönet
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Teyit.org URL'si
url = "https://teyit.org/analiz"
driver.get(url)
time.sleep(3)  # Sayfanın yüklenmesini bekle

# Örnek: haberleri çekmek
articles = driver.find_elements(By.CSS_SELECTOR, "div.feed.d-flex.py-4")

data = []
for article in articles:
    link_tag = article.find_element(By.CSS_SELECTOR, "a.article-link")
    title_tag = article.find_element(By.CSS_SELECTOR, "h2.article-title")
    summary_tag = article.find_elements(By.CSS_SELECTOR, "p.article-summary")

    link = link_tag.get_attribute("href")
    title = title_tag.text.strip()
    summary = summary_tag[0].text.strip() if summary_tag else ""

    data.append({
        "title": title,
        "link": link,
        "summary": summary
    })

# CSV’ye kaydet
df = pd.DataFrame(data)
df.to_csv("data/processed/teyit_fake_news.csv", index=False, encoding="utf-8-sig")

print("Haberler başarıyla çekildi ve CSV’ye kaydedildi!")

driver.quit()
