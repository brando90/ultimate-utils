import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer, PreTrainedModel, PretrainedConfig


class MyTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, **kwargs):
        super().__init__(vocab_file, **kwargs)

    def __call__(self, text):
        tokens = text.split()
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids


class MyModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, **kwargs):
        embeddings = self.embedding(input_ids)
        pooled = torch.mean(embeddings, dim=1)
        pooled = self.dropout(pooled)
        logits = self.linear(pooled)
        return logits


config = PretrainedConfig(vocab_size=1000, hidden_size=128, num_labels=2, hidden_dropout_prob=0.5)
tokenizer = MyTokenizer("path/to/vocab/file")
model = MyModel(config)

input_ids = tokenizer("This is a test")
logits = model(torch.tensor([input_ids]))
