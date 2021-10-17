import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./wrime.tsv", sep="\t")

train_df, test_df = train_test_split(df, test_size=0.20, random_state=0)

train_df.to_csv("./train.tsv", sep="\t", index=False)
test_df.to_csv("./test.tsv", sep="\t", index=False)
