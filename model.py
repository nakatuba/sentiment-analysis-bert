import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input, output_attentions=False):
        if output_attentions:
            _, pooler_output, attentions = self.bert(
                **input, output_attentions=True, return_dict=False
            )
        else:
            _, pooler_output = self.bert(
                **input, output_attentions=False, return_dict=False
            )
        output = self.dropout(pooler_output)
        output = self.linear(output)

        if output_attentions:
            return output, attentions
        else:
            return output
