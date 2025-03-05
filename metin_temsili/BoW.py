# count vectorizer iceriye aktar
from sklearn.feature_extraction.text import CountVectorizer

#veri seti olustur
documents =[
    "kedi bahçede",
    "kedi evde"
]

#vectorizer tanımla 
vectorizer = CountVectorizer()

#metni sayisal vektorlere cevir
X = vectorizer.fit_transform(documents)

#kelime kumesi olusturma
feature_names = vectorizer.get_feature_names_out()
print(f"kelime kumesi : {feature_names}")

#vektör temsili 
vector_temsili = X.toarray()
print(f"vektor temsili : {vector_temsili}")

"""
kelime kumesi : ['bahçede', 'evde', 'kedi']
vektor temsili : 
[[1 0 1]
 [0 1 1]]
"""