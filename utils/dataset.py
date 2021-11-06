import pandas as pd
from torch.utils.data import Dataset


class WrimeDataset(Dataset):
    def __init__(self, path, target, sentiment, binary):
        df = pd.read_csv(path, sep="\t")
        self.texts = df["Sentence"].values

        sentiment = sentiment.capitalize()

        if target == "writer":
            self.labels = df[f"Writer_{sentiment}"]
        elif target == "reader":
            self.labels = df[f"Avg. Readers_{sentiment}"]
        elif target == "gap":
            self.labels = df[f"Writer_{sentiment}"] - df[f"Avg. Readers_{sentiment}"]

        if binary:
            self.labels = (self.labels > 1).astype(int)

        self.labels = self.labels.values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
