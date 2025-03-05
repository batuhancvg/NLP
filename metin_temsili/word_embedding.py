"""
word2text(google)
fasttext(facebook-meta)
"""
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

sentences = [
    "Kopek cok tatli bir hayvandir",
    "Kopekler evcil hayvanlardir",
    "Kediler genellikle bagimsiz hareket etmeyi severler",
    "Hayvanlar insanlarin dostudur",
]

tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
#word2vec
word2_vec_model =  Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)
#fast2vec
fast_text_model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=1)

#gorsellestirme: PCA
def plot_word_embedding(model, tittle):
    
    word_vectors = model.wv
    words = list(word_vectors.key_to_index)[:1000]
    vectors = [word_vectors[word] for word in words]

    #PCA
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)

    #3d gorsellestirme
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    #vektörler
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2])

    #kelimeleri etiketle
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word, fontsize=12)

    ax.set_title(tittle)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_zlabel("component 3")
    plt.show()

plot_word_embedding(word2_vec_model, "Word2Vec")
plot_word_embedding(fast_text_model, "FastText")