"""AlexNet adapted to 32x32 and 28x28, and a dense causal language model.

The 2012 AlexNet first convolution is 11x11 with stride 4, written for a
224x224 image. On a 32x32 CIFAR-10 image that layer collapses the map before
the rest of the network runs. This adaptation keeps five convolution stages
with channel counts 64, 192, 384, 256, 256, uses 3x3 stride-1 convolutions,
and three 2x2 stride-2 pools. An adaptive average pool yields a 4x4 map so
the classifier is Linear(4096) then Linear(4096), which is the AlexNet-scale
fully connected stack. Local response normalization from the 2012 network is
omitted. Dropout on the two hidden classifier layers is 0.5, as in AlexNet.

MNIST uses the same network with one input channel. The pool still ends at
4x4, so the classifier shape matches CIFAR-10.

The language model is a dense causal Transformer: RMSNorm, rotary positions,
and SwiGLU. The token embedding and the output layer share one weight matrix.
It is a Qwen/DeepSeek-style block, at about 100M to 300M parameters, not a
frontier model and not a mixture of experts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNetSmallImage(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x))
        x = torch.flatten(x, 1)
        return self.classifier(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(dtype=x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x is batch, head, time, dim. cos/sin are 1, 1, time, dim.
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, n_head: int):
        super().__init__()
        if d % n_head != 0:
            raise ValueError(f"d={d} is not divisible by n_head={n_head}")
        self.n_head = n_head
        self.head_dim = d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, d: int, n_head: int, hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class DenseCausalLM(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        seq_len: int = 256,
        ffn_mult: float = 8 / 3,
    ):
        super().__init__()
        hidden = int(d * ffn_mult)
        hidden = max(n_head, hidden // 64 * 64)
        self.vocab = vocab
        self.d = d
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, n_head, hidden) for _ in range(n_layer))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.tie_weights()
        cos, sin = _rope_cache(seq_len, d // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def tie_weights(self) -> None:
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        if t > self.seq_len:
            raise ValueError(f"sequence length {t} exceeds {self.seq_len}")
        x = self.tok_emb(idx)
        cos = self.cos[:, :, :t]
        sin = self.sin[:, :, :t]
        for block in self.blocks:
            if x.is_cuda:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, use_reentrant=False
                )
            else:
                x = block(x, cos, sin)
        return self.head(self.norm(x))


def _rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    if head_dim % 2 != 0:
        raise ValueError("rotary head dim must be even")
    pos = torch.arange(seq_len, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freq = torch.outer(pos, inv_freq)
    # Repeat so the rotation matches _rotate_half, which swaps the two halves.
    cos = torch.cos(freq).repeat(1, 2)
    sin = torch.sin(freq).repeat(1, 2)
    return cos.view(1, 1, seq_len, head_dim), sin.view(1, 1, seq_len, head_dim)


def count_parameters(model: nn.Module) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if p.requires_grad:
            total += p.numel()
    return total


def adamw_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    seen = set()
    for p in model.parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def full_lm_config() -> dict:
    return {
        "vocab_placeholder": 50257,
        "d": 768,
        "n_layer": 12,
        "n_head": 12,
        "seq_len": 256,
        "ffn_mult": 8 / 3,
    }
