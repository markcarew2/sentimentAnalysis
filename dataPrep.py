import pandas as pd
import glob

reviews = []

for i in range(12500):
    pathy = glob.glob("aclImdb\\test\\pos\\%s_*" % i)[0]
    with open(pathy, "r", encoding="utf-8") as f:
        text = f.read()
    reviews.append([text, 1])

for i in range(12500):
    pathy = glob.glob("aclImdb\\test\\neg\\%s_*" % i)[0]
    with open(pathy, "r", encoding="utf-8") as f:
        text = f.read()
    reviews.append([text, 0])

df = pd.DataFrame(reviews, columns=["text", "polarity"])

df.to_csv("imdb_te.csv", index_label="row_number")