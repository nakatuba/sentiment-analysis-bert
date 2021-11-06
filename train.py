import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import to_absolute_path
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import BertJapaneseTokenizer

from model import BertClassifier
from utils.dataset import WrimeDataset
from utils.trainer import train


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = WrimeDataset(
        path=to_absolute_path("./data/train.tsv"),
        target=cfg.label.target,
        sentiment=cfg.label.sentiment,
        binary=cfg.label.binary,
    )
    test_dataset = WrimeDataset(
        path=to_absolute_path("./data/test.tsv"),
        target=cfg.label.target,
        sentiment=cfg.label.sentiment,
        binary=cfg.label.binary,
    )

    tokenizer = BertJapaneseTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )

    def collate_batch(batch):
        input_list = tokenizer(
            [text for text, _ in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        label_list = torch.tensor([label for _, label in batch])
        return input_list.to(device), label_list.to(device)

    weights = [
        1 / (train_dataset.labels == label).sum() for label in train_dataset.labels
    ]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        collate_fn=collate_batch,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    num_classes = 2 if cfg.label.binary else 4

    weight = torch.tensor(
        [1 / (train_dataset.labels == label).sum() for label in range(num_classes)],
        dtype=torch.float,
    )

    model = BertClassifier(output_dim=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    num_epochs = cfg.train.num_epochs
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | train | Loss: {train_loss:.4f} Accuracy: {train_acc:.4f}"
        )

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for input, label in test_dataloader:
            output = model(input)
            y_true += label.tolist()
            y_pred += output.argmax(dim=1).tolist()

    print(classification_report(y_true, y_pred))

    torch.save(model.state_dict(), "./model.pt")


if __name__ == "__main__":
    main()
