import pandas as pd
from torch.utils.data import Dataset


class WrimeDataset(Dataset):
    def __init__(self, split, label):
        df = pd.read_csv("./data/wrime.tsv", sep="\t")
        self.texts = df[df["Train/Dev/Test"] == split]["Sentence"].values
        self.labels = df[df["Train/Dev/Test"] == split][label].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
