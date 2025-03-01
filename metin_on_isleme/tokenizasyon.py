import nltk #natural language toolkit

nltk.download('punkt') # metni kelime ve cümle bazinda tokenlara ayirir

text = "Merhaba, benim adım Ahmet. Bugün hava çok güzel. Siz nasılsınız?"

#kelime tokenizasyonu: metni kelimelere ayirir
#noktalama işaretleri ve bosluklar ayri birer token olarak ele alinir
word_tokens = nltk.word_tokenize(text)
print(word_tokens)

#cumle tokenizasyonu: metni cumlelere ayirir, her bir cümle bir token olarak ele alinir
sentence_tokens = nltk.sent_tokenize(text)
print(sentence_tokens)

