import pandas as pd
from torch.utils.data import Dataset


class WrimeDataset(Dataset):
    def __init__(self, path, label, binary=True):
        df = pd.read_csv(path, sep="\t")
        self.texts = df["Sentence"].values

        if label == "writer":
            self.labels = df["Writer_Anger"]
        elif label == "reader":
            self.labels = df["Avg. Readers_Anger"]
        elif label == "gap":
            self.labels = df["Writer_Anger"] - df["Avg. Readers_Anger"]

        if binary:
            self.labels = (self.labels > 1).astype(int)

        self.labels = self.labels.values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
