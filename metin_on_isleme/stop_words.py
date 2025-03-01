#ing stop words analizi (nltk)
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') #stop words analizi icin gerekli veri tabanı
stop_words_eng = set(stopwords.words("english")) #ingilizce stop words listesi
print(f"stop words: {stop_words_eng}")

text  = "There are some examples of handling stop words from some texts."
text_list = text.split()
filtered_words = [word for word in text_list if word.lower() not in stop_words_eng]
print(f"filtered words: {filtered_words}")

stop_words_tr = set (stopwords.words("turkish")) #turkce stop words listesi
print(f"stop words: {stop_words_tr}")
metin ="merhaba arkadaslar cok guzel bir ders isliyoruz."
metin_list = metin.split()
filtered_words= [ word for word in metin_list if word.lower() not in stop_words_tr]
print(f"filtered words: {filtered_words}")

#kutuphanesiz stop words cikarimi 
tr_stopwords = ["için","bu", "ile","mu","mi","özel"]

metin = "Bu bir denemedir. Amacimiz bu metinde bulunan özel karakterleri elemek mi acaba"
filtered_words = [ word for word in metin.split() if word.lower() not in tr_stopwords]
print(f"filtered words: {filtered_words}")