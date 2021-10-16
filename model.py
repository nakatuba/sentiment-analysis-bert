import torch.nn as nn
from transformers import BertModel


class Bert(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input):
        output = self.bert(**input)["pooler_output"]
        output = self.dropout(output)
        output = self.linear(output)
        return output


class BertForMultiTask(nn.Module):
    def __init__(self, num_labels, num_tasks):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        self.dropout = nn.Dropout(0.1)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.bert.config.hidden_size, num_labels)
                for _ in range(num_tasks)
            ]
        )

    def forward(self, input):
        output = self.bert(**input)["pooler_output"]
        output = self.dropout(output)
        output = [linear(output) for linear in self.linears]
        return output
