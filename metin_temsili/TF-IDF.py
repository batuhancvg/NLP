import pandas as pd 
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

#örnek dökümanlar
documents = ["kopek cok tatli hayvanlardir",
             "kppek ve kuslar cok tatli hayvanlardir",
             "Inekler sUt uretirler"]

# vectörizer tanımla 
tfidf_vectorizer = TfidfVectorizer()

#metinleri sayısal verilere dönüştür
X = tfidf_vectorizer.fit_transform(documents)

#kelime kümseini incele
feature_nnames = tfidf_vectorizer.get_feature_names_out()
print(feature_nnames)

#vektör temsili incele
vektör_temsili = X.toarray() 
print(f"tf-idf vektör temsili: \n{vektör_temsili}")
df_tf_idf = pd.DataFrame(vektör_temsili, columns=feature_nnames)   

#ortalama tf-idf değerlerine bakalım
tf_idf = df_tf_idf.mean(axis=0)
print(f"ortalama tf-idf değerleri: \n{tf_idf}")


