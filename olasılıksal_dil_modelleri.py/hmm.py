#  import labraries
import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000

# gerekli veri setini içeri aktar
nltk.download('conll2000')

train_data = conll2000.tagged_sents('train.txt')
test_data = conll2000.tagged_sents('test.txt')

# train hmm
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# yeni cumle ve test
test_sentence = "I like going to school".split()
tags=hmm_tagger.tag(test_sentence)   
print(f"Tagged sentence: {tags}")
# Tagged sentence: [('I', 'PRP'), ('like', 'IN'), ('going', 'VBG'), ('to', 'TO'), ('school', 'NN')]