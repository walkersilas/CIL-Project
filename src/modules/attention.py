"""
    Attention for Collaborative Filtering

    Adopted from https://github.com/Darel13712/acf_pytorch/blob/master/nets.py
    based on the paper https://ai.tencent.com/ailab/media/publications/Wei_Liu-Attentive_Collaborative_Filtering_Multimedia_Recommendation-SIGIR17.pdf
"""

import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax


def _kaiming_(layer):
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    torch.nn.init.zeros_(layer.bias)


class Attention(nn.Module):
    """Get user embedding accounting to surpassed items"""

    def __init__(self, user_count, item_count, embedding_size=128, device=None):
        super(Attention, self).__init__()

        self.embedding_size = embedding_size

        self.user_embedding = nn.Embedding(user_count + 1, embedding_size)
        self.item_embedding = nn.Embedding(item_count + 1, embedding_size)

        self.linear1 = nn.Linear(embedding_size * 2, embedding_size)
        _kaiming_(self.linear1)

        self.linear2 = nn.Linear(embedding_size, 1)
        _kaiming_(self.linear2)

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.device = device

    def forward(self, user_id, item_ids):
        user = self.user_embedding(user_id)
        items = self.item_embedding(item_ids)

        user_expanded = user.expand(items.shape[0], -1)
        components = torch.tensor([], device=self.device)
        output = torch.cat([user_expanded, items, components], dim=-1)

        output = self.linear1(output)
        output = relu(output)

        output = self.linear2(output)
        output = softmax(output, 0)

        output = user + (items * output).sum(0)

        return output

    def score(self, user, items):
        return (user * items).sum(1) / self.embedding_size

    @property
    def params(self):
        return [param for _, param in self.named_parameters() if param.requires_grad == True]
