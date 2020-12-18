import pandas as pd
import re

df = pd.read_csv("imdb_tr2.csv", index_col = 0)

text = df.iloc[0,0]

words = re.