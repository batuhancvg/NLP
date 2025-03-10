#import libaries
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.util import ngrams #for n-gram
from nltk.tokenize import word_tokenize #for tokenization

from collections import Counter #for counting

"""
kelimeler = ['elma', 'armut', 'elma', 'kiraz', 'elma', 'armut']
sayac = Counter(kelimeler)
print(sayac)
Counter({'elma': 3, 'armut': 2, 'kiraz': 1})
"""

#example data set
corpus = [
    "I love apple",
    "I love him",
    "I love NLP",
    "You love me",
    "He loves apple",
    "I love you and you love me"
]

"""
problem tanimla:
-dil modeli yapmak
-amac 1 kelimeden sonra gelecek kelimeyi tahmin etmek: metin turetmek/olusturmak.
-bunun icin n gram modeli kullanilir.

ex: I ....(love).... apple
"""
#dataset tokenization
tokens = [word_tokenize(sentences.lower()) for sentences in corpus]
print(f"Tokenized Data: {tokens}")

# crete bigram model
bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))
print(f"Bigram: {bigrams}")

bigrams_freq = Counter(bigrams)
print(f"Bigram Frequency: {bigrams_freq}")

trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list,3)))
print(f"Trigram: {trigrams}")

trigrams_freq = Counter(trigrams)
print(f"Trigram Frequency: {trigrams_freq}")

#model testing
# ı love bigram dan sonra you veya apple kelimelerinin gelme olasılıgı

bigram = ("i","love")
prob_you = trigrams_freq[("i","love","you")]/bigrams_freq[bigram]
print(f"Probability of 'you' after 'I love': {prob_you}")

prob_apple = trigrams_freq[("i","love","apple")]/bigrams_freq[bigram]
print(f"Probability of 'apple' after 'I love': {prob_apple}")