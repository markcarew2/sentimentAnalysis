import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.kernel_approximation import Nystroem, RBFSampler


df = pd.read_csv("imdb_tr.csv", index_col = 0)
testData = pd.read_csv("imdb_te.csv", index_col=0)

unigramVectorizer = CountVectorizer(stop_words={"english"})

unigramVector = unigramVectorizer.fit_transform(df["text"])
unigramVocab = unigramVectorizer.get_feature_names()


#print(len(unibigramVectorizer.get_feature_names()))

clfUnigram = SGDClassifier(loss="log", penalty="l1")
clfUnigram.fit(unigramVector, y = df["polarity"])

testVector = unigramVectorizer.transform(testData["text"])
pred = clfUnigram.predict(testVector)

acc = metrics.accuracy_score(testData["polarity"], pred)
F1Score = metrics.f1_score(testData["polarity"], pred, average="macro")

print(acc)
print(F1Score)
print("*****")

unibigramVectorizer = CountVectorizer(ngram_range=(1,2), stop_words={"english"})

bigramVector = unibigramVectorizer.fit_transform(df["text"], y = df["polarity"])
clfBigram = SGDClassifier(loss="log", penalty="l1")
clfBigram.fit(bigramVector,y = df["polarity"])

"""
sgdNystroem = SGDClassifier(loss="log", penalty="l1", learning_rate="constant", eta0=1)
feature_map_nystroem = Nystroem(gamma=1,random_state=1, n_components=100)
data_transformed = feature_map_nystroem.fit_transform(bigramVector)
print(data_transformed.shape)
sgdNystroem.fit(data_transformed, df["polarity"])
print(sgdNystroem.score(data_transformed, df["polarity"]))
"""

testVector = unibigramVectorizer.transform(testData["text"])

pred = clfBigram.predict(testVector)
acc = metrics.accuracy_score(testData["polarity"], pred)
F1Score = metrics.f1_score(testData["polarity"], pred, average="macro")

"""
nystroemTest = feature_map_nystroem.transform(testVector)

pred = sgdNystroem.predict(nystroemTest)
acc1 = metrics.accuracy_score(testData["polarity"], pred)
F1Score1 = metrics.f1_score(testData["polarity"], pred, average="macro")
"""

print("SGD Accuracy: " ,acc)
print("SGD F1: ", F1Score)

"""
print("******")
print("Nystroem Accuracy: " ,acc1)
print("Nystroem F1: ", F1Score1)
print("******")
"""