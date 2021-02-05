import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.feature_selection import SelectKBest, chi2
import time

startTime = time.time()
df = pd.read_csv("imdb_tr.csv", index_col = 0)
testData = pd.read_csv("imdb_te.csv", index_col=0)


#Train with unigram and reduced unigram, chose approx. 1% of features

#Want to measure this time to add to the Reduced Unigram Time 
#get approx time it would take reduced Unigram to do this without having to run twice
unigramConstructTimeStart = time.time()
"""
unigramVectorizer = CountVectorizer()
unigramVector = unigramVectorizer.fit_transform(df["text"])
clfUnigram = SGDClassifier(loss="log", penalty="l1")
"""
unigramConstructTime = time.time() - unigramConstructTimeStart
"""
clfUnigram.fit(unigramVector, y = df["polarity"])
trainScore1 = clfUnigram.predict(unigramVector)
trainScore = metrics.accuracy_score(df["polarity"], trainScore1)
print("Not reduced Unigram Train Score: ", trainScore)

testVector = unigramVectorizer.transform(testData["text"])
pred = clfUnigram.predict(testVector)

acc = metrics.accuracy_score(testData["polarity"], pred)
F1Score = metrics.f1_score(testData["polarity"], pred, average="macro")

print("NonReduced Unigram Accuracy/F1: ", acc)
print(F1Score)
"""
unigramTime = time.time() - startTime

"""
print("Unigram Time = %s seconds" % unigramTime)
print("*****")


#Reduced Unigram
selectBest = SelectKBest(chi2, k =1000)
unigramVectorReduce = selectBest.fit_transform(unigramVector, df["polarity"])

clfUnigram.fit(unigramVectorReduce, y = df["polarity"])

trainScore1 = clfUnigram.predict(unigramVectorReduce)
trainScore = metrics.accuracy_score(df["polarity"], trainScore1)
print("Reduced Unigram Training Accuracy: ", trainScore)

testVectorReduce = selectBest.transform(testVector)
pred = clfUnigram.predict(testVectorReduce)
acc = metrics.accuracy_score(testData["polarity"], pred)
F1Score = metrics.f1_score(testData["polarity"], pred, average="macro")


print("Reduced Unigram Accuracy/F1: ", acc)
print(F1Score)
"""
unigramReducedTime = time.time() - unigramTime - startTime + unigramConstructTime

"""
print("Unigram Reduced Time = %s seconds" % unigramReducedTime)
print("*****")
"""


#Try again with Bigrams, reduce to about 1%
bigramConstructTimeStart = time.time()
unibigramVectorizer = CountVectorizer(ngram_range=(1,2))
bigramVector = unibigramVectorizer.fit_transform(df["text"], y = df["polarity"])
clfBigram = SGDClassifier(loss="hinge", penalty="l2", alpha=.01, max_iter=10000)
bigramConstructTime = time.time() - bigramConstructTimeStart

clfBigram.fit(bigramVector,y = df["polarity"])

trainScore1 = clfBigram.predict(bigramVector)
trainScore = metrics.accuracy_score(df["polarity"], trainScore1)
print("NonReduced Bigram Training Accuracy: ", trainScore)


testVector = unibigramVectorizer.transform(testData["text"])

pred = clfBigram.predict(testVector)
acc = metrics.accuracy_score(testData["polarity"], pred)
F1Score = metrics.f1_score(testData["polarity"], pred, average="macro")

print("Bigram NonReduced Accuracy/F1: " ,acc)
print(F1Score)
bigramTime = time.time() - unigramTime - unigramReducedTime - startTime
#print("Bigram Time = %s seconds" % bigramTime)
print("******")

#Reduced Bigram
selectBestBigram = SelectKBest(chi2, k =15138)
bigramVectorReduce = selectBestBigram.fit_transform(bigramVector, df["polarity"])
clfBigram.fit(bigramVectorReduce,y = df["polarity"])

trainScore1 = clfBigram.predict(bigramVectorReduce)
trainScore = metrics.accuracy_score(df["polarity"], trainScore1)
print("Reduced Bigram Training Accuracy: ", trainScore)


testVectorReduce = selectBestBigram.transform(testVector)

pred = clfBigram.predict(testVectorReduce)
acc = metrics.accuracy_score(testData["polarity"], pred)
F1Score = metrics.f1_score(testData["polarity"], pred, average="macro")

print("Bigram Reduced Accuracy/F1: " ,acc)
print(F1Score)
bigramReducedTime = time.time() - bigramTime - unigramTime - unigramReducedTime - startTime + bigramConstructTime
#print("Bigram Reduced Time = %s seconds" % bigramReducedTime)