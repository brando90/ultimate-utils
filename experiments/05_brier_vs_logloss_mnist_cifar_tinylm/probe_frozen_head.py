"""Train only the last layer on frozen MNIST features.

The full network can move the representation and change a logit even when that
class's own softmax gradient is tiny. This probe freezes the convolutional
features and trains the linear head, so a bias that starts at -12 can move
only through that head's gradient.

Prediction, written before this run: with bias -12 and no weight decay, after
2 epochs the log-loss head has class-0 recall above 0.50 and a mean class-0
logit above -2. The Brier head stays below class-0 recall 0.20 and its mean
class-0 logit stays below -8.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from train_compare import LOSSES, SmallCNN, param_groups, set_collapsed_bias, vision_metrics

OUT = Path(__file__).resolve().parent / "results" / "frozen_head.json"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    tfm = transforms.ToTensor()
    root = Path(__file__).resolve().parent / "results" / "data"
    train_set = datasets.MNIST(str(root), train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(str(root), train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    feature_model = SmallCNN(1).to(device)
    feature_model.eval()
    for param in feature_model.features.parameters():
        param.requires_grad_(False)

    @torch.no_grad()
    def embed(loader):
        xs, ys = [], []
        for x, y in loader:
            h = feature_model.features(x.to(device)).flatten(1)
            xs.append(h.cpu())
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    train_h, train_y = embed(train_loader)
    test_h, test_y = embed(test_loader)
    rows = []
    for loss_name in ("log_loss", "brier"):
        for bias in (0.0, -12.0):
            torch.manual_seed(0)
            head = torch.nn.Linear(train_h.shape[1], 10).to(device)
            if bias != 0.0:
                set_collapsed_bias(head, 0, bias)
            opt = torch.optim.AdamW(param_groups(head, 0.0), lr=1e-3)
            fn = LOSSES[loss_name]
            with torch.no_grad():
                start_logit = float(head(train_h[:1024].to(device))[:, 0].mean())
            for epoch in range(2):
                head.train()
                perm = torch.randperm(train_h.shape[0])
                for i in range(0, perm.numel() - 128, 128):
                    batch = perm[i : i + 128]
                    opt.zero_grad(set_to_none=True)
                    logits = head(train_h[batch].to(device))
                    loss = fn(logits, train_y[batch].to(device))
                    loss.backward()
                    opt.step()
            head.eval()

            class Wrap(torch.nn.Module):
                def forward(self, x):
                    return head(feature_model.features(x).flatten(1))

            metrics = vision_metrics(Wrap(), test_loader, device, collapsed=0)
            row = {"loss": loss_name, "collapsed_bias": bias, "start_mean_logit": start_logit, **metrics}
            rows.append(row)
            print(row, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
