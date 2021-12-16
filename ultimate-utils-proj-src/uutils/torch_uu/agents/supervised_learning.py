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

    def eval_forward(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        # - to make sure we get the [B] tensor to compute confidence intervals/error bars
        original_reduction: str = self.loss.reduction
        self.loss.reduction = 'none'
        # - forward
        loss: Tensor = self.loss(self.model(batch))
        acc: Tensor =
        #
        self.loss.reduction = original_reduction
        assert loss.size() == torch.Size([B])
        return loss, acc
