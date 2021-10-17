import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, mean_absolute_error
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import BertJapaneseTokenizer

from model import Bert
from utils.dataset import WrimeDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = WrimeDataset(split="train", label="Avg. Readers_Anger")
    valid_dataset = WrimeDataset(split="dev", label="Avg. Readers_Anger")
    test_dataset = WrimeDataset(split="test", label="Avg. Readers_Anger")

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

    batch_size = 32

    weights = [
        1 / (train_dataset.labels == label).sum() for label in train_dataset.labels
    ]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_batch
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    model = Bert(num_labels=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    num_epochs = 3
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
        valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion)
        print(f"Epoch {epoch + 1}/{num_epochs}", end=" ")
        print(f"| train | Loss: {train_loss:.4f} Accuracy: {train_acc:.4f}", end=" ")
        print(f"| valid | Loss: {valid_loss:.4f} Accuracy: {valid_acc:.4f}")

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for input, label in test_dataloader:
            output = model(input)
            y_true += label.tolist()
            y_pred += output.argmax(dim=1).tolist()

    print(classification_report(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))


def train(model, dataloader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for input, label in dataloader:
        output = model(input)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        acc = (pred == label).sum() / len(pred)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for input, label in dataloader:
            output = model(input)

            loss = criterion(output, label)

            pred = output.argmax(dim=1)
            acc = (pred == label).sum() / len(pred)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


if __name__ == "__main__":
    main()
