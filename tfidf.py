import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics

df = pd.read_csv("imdb_tr.csv", index_col = 0)
testData = pd.read_csv("imdb_te.csv", index_col=0)

uniVectorize = TfidfVectorizer(stop_words = {'english'})
uniVector = uniVectorize.fit_transform(df["text"])
testVector = uniVectorize.transform(testData["text"])

uniclf = SGDClassifier(loss="hinge", penalty="l1")
uniclf.fit(uniVector, df["polarity"])

pred = uniclf.predict(testVector)

acc = metrics.accuracy_score(testData["polarity"], pred)
f1 = metrics.f1_score(testData["polarity"], pred, average="macro")

print(acc)
print(f1)
print("*****")

biVectorize = TfidfVectorizer(stop_words={"english"},ngram_range=(1,2))
biVector = biVectorize.fit_transform(df["text"], y=df["polarity"])
testVector = biVectorize.transform(testData["text"])

biclf = SGDClassifier(loss="hinge", penalty="l1")
biclf.fit(biVector, y = df["polarity"])

pred = biclf.predict(testVector)

acc = metrics.accuracy_score(testData["polarity"], pred)
f1 = metrics.f1_score(testData["polarity"], pred, average="macro")

print(acc)
print(f1)