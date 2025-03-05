import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from gensim.models import Word2Vec 
from gensim.utils import simple_preprocess

#veri seti yükleme
df = pd.read_csv(r"metin_temsili\IMDB Dataset.csv")
documents = df["review"]

#veri seti temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join([word for word in text.split() if len(word) > 2])
    text = simple_preprocess(text)
    return text

# celan = [clean_text("ASDASAD 1551 %& I merhabA")]
# print(celan)
cleaned_documents = [clean_text(doc) for doc in documents]

#metin tokenization
tokenized_documents = cleaned_documents

#word2vec modeli oluşturma
model = Word2Vec(sentences=tokenized_documents, vector_size=50, window=5, min_count=1, sg=0)
word_vectors = model.wv

word = list(word_vectors.key_to_index.keys()) [:500]
vectors = [word_vectors[word] for word in word]

#clustering KMeans K = 2
kmeans = KMeans(n_clusters=2)
kmeans.fit(vectors)
clusters = kmeans.labels_

#PCA 50 -> 2 
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

#2 boyutlu görselleştirme
plt.figure()
plt.scatter(reduced_vectors[:,0], reduced_vectors[:,1], c=clusters, cmap="viridis")

centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c="red", marker="x", s=150, label= "Center")
plt.legend()

#figure üzerine kelime yazdırma
for i , word in enumerate(word):
    plt.text(reduced_vectors[i,0], reduced_vectors[i,1], word, fontsize=7)

plt.title("Word2Vec Word Embedding")
plt.show()