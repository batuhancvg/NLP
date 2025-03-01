import nltk

nltk.download('wordnet') #wordnet : lemmazitation islemi icin gerekli veri tabanı   

from nltk.stem import PorterStemmer #stemming islemi icin fonksiyon

#porter stemmer nesnesini olusturuyoruz
stemmer = PorterStemmer()
words = ["runner", "running", "ran", "runs", "better", "go","went"]
stems = [stemmer.stem(word) for word in words] #stemming islemi porter stemmer ile yapılır stem(kök) alır
print(f"stems: {stems}") 

from nltk.stem import WordNetLemmatizer #lemmatization islemi icin fonksiyon

lemmatizer = WordNetLemmatizer()
words = ["runner", "running", "ran", "runs", "better", "go","went"]
lemmas = [lemmatizer.lemmatize(w, pos="v") for w in words] #lemmatization islemi yapılır lemmatizer anlamlı kök alır
print(f"lemmas: {lemmas}")


