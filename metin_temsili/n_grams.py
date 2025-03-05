#import libraries
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Bu calisma NGram calismasidir.",
    "Bu calisma dogal dil isleme calismasidir."
]

vectorizer_unigram = CountVectorizer(ngram_range=(1,1)) #unigram
vectorizer_bigram = CountVectorizer(ngram_range=(2,2)) #bigram
vectorizer_trigram = CountVectorizer(ngram_range=(3,3)) #trigram   

#unigram 
X_unigram = vectorizer_unigram.fit_transform(documents)
feature_names_unigram = vectorizer_unigram.get_feature_names_out()
print(f"Unigram: {feature_names_unigram}")
#bigram
X_bigram = vectorizer_bigram.fit_transform(documents)
feature_names_bigram = vectorizer_bigram.get_feature_names_out()
print(f"Bigram: {feature_names_bigram}")
#trigram    
X_trigram = vectorizer_trigram.fit_transform(documents)
feature_names_trigram = vectorizer_trigram.get_feature_names_out() 
print(f"triigram: {feature_names_trigram}") 