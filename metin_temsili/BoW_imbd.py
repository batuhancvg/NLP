# kutuphaneleri ice aktarma 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import re 
from collections import Counter

#veri setini iceriye aktarma 
df = pd.read_csv(r"C:\Users\Tiger\Desktop\NLP\metin_temsili\IMDB Dataset.csv")

#metin verilerini alalım
documents = df["review"]
labels =  df["sentiment"] #positive veya negative

#metin temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join (word for word in text.split() if len(word) > 2)
    return text 

#metinleri temizle
cleaned_doc = [clean_text(row) for row in documents]

#BoW
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(cleaned_doc [:75]) 
feature_names = vectorizer.get_feature_names_out()

vektor_temsili2 = x.toarray()
# print(f"vektör temsili: {vektor_temsili2}")

df_BoW = pd.DataFrame(vektor_temsili2, columns = feature_names)
word_counts = x.sum(axis = 0).A1
word_freq = dict(zip(feature_names, word_counts))

most_common_5_words = Counter(word_counts).most_common(5)
print(f"En cok kullanilan 5 kelime: {most_common_5_words}")



