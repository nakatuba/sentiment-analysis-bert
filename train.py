import os

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import BertJapaneseTokenizer

from model import BertClassifier
from utils.dataset import WrimeDataset
from utils.trainer import train


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = WrimeDataset(
        path=to_absolute_path(cfg.data.train_path),
        target=cfg.label.target,
        sentiment=cfg.label.sentiment,
        num_classes=cfg.label.num_classes,
    )
    test_dataset = WrimeDataset(
        path=to_absolute_path(cfg.data.test_path),
        target=cfg.label.target,
        sentiment=cfg.label.sentiment,
        num_classes=cfg.label.num_classes,
    )

    tokenizer = BertJapaneseTokenizer.from_pretrained(cfg.model.pretrained_model)

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

    weight = torch.tensor(
        [
            1 / (train_dataset.labels == label).sum()
            for label in range(cfg.label.num_classes)
        ],
        dtype=torch.float,
    )

    model = BertClassifier(
        pretrained_model=cfg.model.pretrained_model,
        output_dim=cfg.label.num_classes,
    ).to(device)
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
    y_score = []

    with torch.no_grad():
        for input, label in test_dataloader:
            output = model(input)
            y_true += label.tolist()
            y_pred += output.argmax(dim=1).tolist()
            y_score += output[:, 1].tolist()

    print(classification_report(y_true, y_pred, digits=3))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.rcParams["font.size"] = 20
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.legend(loc="lower right")
    plt.tight_layout()

    wandb.log({"ROC curve": wandb.Image(plt), "AUC": auc})

    df = pd.read_csv(to_absolute_path(cfg.data.test_path), sep="\t")
    df["GT"] = y_true
    df["Predicted"] = y_pred
    df.to_csv(os.path.join(wandb.run.dir, "result.tsv"), sep="\t", index=False)

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))


if __name__ == "__main__":
    main()
