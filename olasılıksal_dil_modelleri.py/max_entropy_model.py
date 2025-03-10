"""
classification problem: duygu analizi -> olumlu veya olumsuz olarak sınıflandırma
"""
# import libraries
from nltk.classify import MaxentClassifier

# veir seti tanımlama
train_data = [
    ({"Love":True, "amazing":True, "happy":True, "terrible":False}, "positive"),
    ({"hate":True, "terrrible":True}, "negative"),
    ({"joy":True, "happy":True, "hate":False}, "positive"),
    ({"sad":True, "depressed": True,"Love":False},"negative")
]

# trainer maximum entropy classifier
classifier = MaxentClassifier.train(train_data, max_iter=10)

# yeni cümle ile test
test_sentece = "I like this movie and it was amazing" 
test_sentece1 = "I hate this movie and it was terrible"
features = {word: (word in test_sentece.lower().split()) for word in ["Love", "amazing", "happy", "terrible", "hate", "terrrible", "joy", "sad", "depressed"]}
features1 = {word: (word in test_sentece1.lower().split()) for word in ["Love", "amazing", "happy", "terrible", "hate", "terrrible", "joy", "sad", "depressed"]}
classifier.classify(features)
classifier.classify(features1)
print(f"result: {classifier.classify(features)}") # result: positive
print(f"result: {classifier.classify(features1)}") # result: negative


