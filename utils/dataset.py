import pandas as pd
from torch.utils.data import Dataset


class WrimeDataset(Dataset):
    def __init__(self, path, label):
        df = pd.read_csv(path, sep="\t")
        self.texts = df["Sentence"].values
        self.labels = df[label].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
