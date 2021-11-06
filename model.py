import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input, output_attentions=False):
        _, pooler_output, attentions = self.bert(
            **input, output_attentions=True, return_dict=False
        )
        output = self.dropout(pooler_output)
        output = self.linear(output)

        return (output, attentions) if output_attentions else output
