import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(r"C:\Users\Tiger\Desktop\NLP\metin_temsili\spam.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.text)
feature_names = vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis=0).A1

df_tfidf = pd.DataFrame("word:", feature_names, "tfidf_score:", tfidf_score)

df_tfidf_sorted = df.tfidf.sort_values(ascending=False,asceding=False)
print(df_tfidf_sorted.head(10))