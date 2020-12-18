import pandas as pd

df = pd.read_csv("imdb_tr.csv", index_col=0)

with open("stopwords.txt", "r") as f:
    a = f.readlines()

b = []
for thing in a:
    b.append(thing.strip())

for word in b:
    df["text"] = df["text"].apply(lambda x: x.lower().replace(" " + word + " ", " "))
    df["text"] = df["text"].apply(lambda x: x.replace(" " + word + ".", " "))
    df["text"] = df["text"].apply(lambda x: x.replace(" " + word + ",", " "))
    df["text"] = df["text"].apply(lambda x: x.replace(" " + word + "?", " "))
    df["text"] = df["text"].apply(lambda x: x.replace(" " + word + ":", " "))
    df["text"] = df["text"].apply(lambda x: x.replace(" " + word + ";", " "))
    df["text"] = df["text"].apply(lambda x: x.replace(" " + word + r"\"", " "))
    df["text"] = df["text"].apply(lambda x: x.replace(" " + word + r"\'", " "))

df["text"] = df["text"].apply(lambda x: x.replace(r"<br /><br />", " "))
df.to_csv("imdb_tr2.csv")