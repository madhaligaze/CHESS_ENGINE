"""Human move model: a small SE-ResNet over the 8x8 board.

Policy: every square gets a query and a key vector; the logit of move (from, to) is
q[from] . k[to] / sqrt(d) plus a learned per-(from, to) bias. Value: win/draw/loss logits.
The Rust engine re-implements exactly this forward pass (src/nn.rs).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NUM_PLANES


class SqueezeExcite(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 2 * channels)

    def forward(self, x):
        s = self.fc2(F.relu(self.fc1(x.mean(dim=(2, 3)))))
        scale, bias = s.chunk(2, dim=1)
        return x * torch.sigmoid(scale)[:, :, None, None] + bias[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, channels, se_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcite(channels, se_hidden)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.se(self.bn2(self.conv2(y)))
        return F.relu(x + y)


class HumanNet(nn.Module):
    def __init__(self, channels=64, blocks=6, se=16, policy_dim=32, value_channels=8, value_hidden=128):
        super().__init__()
        self.cfg = dict(
            channels=channels,
            blocks=blocks,
            se=se,
            policy_dim=policy_dim,
            value_channels=value_channels,
            value_hidden=value_hidden,
        )
        self.stem = nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResBlock(channels, se) for _ in range(blocks)])
        self.pq = nn.Linear(channels, policy_dim)
        self.pk = nn.Linear(channels, policy_dim)
        self.pbias = nn.Parameter(torch.zeros(64 * 64))
        self.vconv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.vbn = nn.BatchNorm2d(value_channels)
        self.vfc1 = nn.Linear(value_channels * 64, value_hidden)
        self.vfc2 = nn.Linear(value_hidden, 3)

    def forward(self, x):
        h = F.relu(self.stem_bn(self.stem(x)))
        for block in self.blocks:
            h = block(h)
        t = h.flatten(2).transpose(1, 2)  # (B, 64 squares, C)
        q, k = self.pq(t), self.pk(t)
        logits = torch.bmm(q, k.transpose(1, 2)) * (1.0 / math.sqrt(q.shape[-1]))
        logits = logits.flatten(1) + self.pbias  # index = from * 64 + to
        v = F.relu(self.vbn(self.vconv(h))).flatten(1)
        v = self.vfc2(F.relu(self.vfc1(v)))
        return logits, v


def masked_policy_loss(logits, mask, target):
    logits = logits.masked_fill(~mask, -1e9)
    return F.cross_entropy(logits, target), logits
