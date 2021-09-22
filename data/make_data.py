import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./wrime.tsv", sep="\t")

df = df[
    [
        "Sentence",
        "Writer_Joy",
        "Reader1_Joy",
        "Reader2_Joy",
        "Reader3_Joy",
    ]
]

df["Sentence"] = df["Sentence"].replace(r"\\n", "", regex=True)
df["Readers_mean"] = df[["Reader1_Joy", "Reader2_Joy", "Reader3_Joy"]].mean(axis=1)
df["Readers_std"] = df[["Reader1_Joy", "Reader2_Joy", "Reader3_Joy"]].std(axis=1)
df["Difference"] = df["Writer_Joy"] - df["Readers_mean"]

df["Label"] = 0
df.loc[df["Readers_std"] < 1, "Label"] = 1
df.loc[(df["Readers_std"] < 1) & (df["Difference"] > 1), "Label"] = 2
df.loc[(df["Readers_std"] < 1) & (df["Difference"] < -1), "Label"] = 3

train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=0, stratify=df["Label"]
)

df.to_csv("./joy.tsv", sep="\t", index=False)
train_df.to_csv("./train.tsv", sep="\t", index=False)
test_df.to_csv("./test.tsv", sep="\t", index=False)
