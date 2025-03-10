#part of speech  POS: kelimelerin uygun sozcuk turunu bulma calismasi
#  I (zamir) am a teacher (isim)

import nltk 
from nltk.tag import hmm 


# ornek training cumlesi
train_data = [
    [("I", "PRP"), ("am", "VBP"), ("a", "DT"), ("teacher", "NN")],
    [("You", "PRP"), ("are", "VBP"), ("a", "DT"), ("student", "NN")]
]

# train HMM 
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# yeni bir cumle olustur ve cumle icerisinde bulunan her bir sozcgun turunu etiketle
test_sentence = "I am a student".split() # liste haline getir 
test_sentence1 = "He is a driver".split() # liste haline getir
tags = hmm_tagger.tag(test_sentence)
tags1 = hmm_tagger.tag(test_sentence1)
print(f"Tagged sentence: {tags}") 
print(f"Tagged sentence: {tags1}")
# Tagged sentence: [('I', 'PRP'), ('am', 'VBP'), ('a', 'DT'), ('student', 'NN')] dogru etiketleme
# Tagged sentence: [('He', 'PRP'), ('is', 'PRP'), ('a', 'PRP'), ('driver', 'PRP')] yanlis etiketleme cunku train_data icerisinde bu cumle yoktu.
# dogru etiketleme icin daha fazla train_data eklemek gerekiyor.



