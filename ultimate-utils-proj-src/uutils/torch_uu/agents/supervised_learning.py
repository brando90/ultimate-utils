from torch import nn, Tensor


class ClassificationSLAgent(nn.Module):

    def __init__(self, model: nn.Module, loss: nn.Module):
        self.model = model
        self.loss = loss
        self.acc =

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        loss: Tensor = self.loss(self.model(batch))
        acc: Tensor =
        return loss, acc