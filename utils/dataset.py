import pandas as pd
from torch.utils.data import Dataset


class WrimeDataset(Dataset):
    def __init__(self, path, target, sentiment, num_classes):
        df = pd.read_csv(path, sep="\t")
        df["Sentence"] = df["Sentence"].replace(r"\\n", "", regex=True)
        self.texts = df["Sentence"].values

        sentiment = sentiment.capitalize()

        if target == "writer":
            self.labels = df[f"Writer_{sentiment}"].values
        elif target == "reader":
            self.labels = df[f"Avg. Readers_{sentiment}"].values
        elif target == "gap":
            self.labels = (
                df[f"Writer_{sentiment}"] - df[f"Avg. Readers_{sentiment}"]
            ).values

        if num_classes == 2:
            self.labels = (self.labels > 1).astype(int)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
