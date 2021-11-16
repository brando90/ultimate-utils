"""
Hyper Basis Function (HBFs) Networks

Ref:
    - https://dspace.mit.edu/handle/1721.1/113159


todo:
    - ask or get norm 2/dist with direct pytorch with a batch x=[B, D], w=[D, D1]
    - ask or get mahalobis distance direct pytorch with a batch x=[B, D]
    - HBF class layer
    - create hierarchical hbfs
    - hp sweep for FCNN then HHBF
"""
import torch
from torch import nn, Tensor

def get_normal_tensor(loc: float, scale: float, B: int, Din: int) -> Tensor:
    """
    Warning: allows optimization of stochastic process https://pytorch.org/docs/stable/distributions.html
    """
    data: torch.Tensor = torch.distributions.Normal(loc=loc, scale=scale).sample((B, Din))
    return data

def get_uniform_tensor(lb: float, ub: float, B: int, Din: int) -> Tensor:
    """
    Warning: allows optimization of stochastic process https://pytorch.org/docs/stable/distributions.html
    """
    data: torch.Tensor = torch.distributions.Uniform(low=lb, high=ub).sample((B, Din))
    return data

class L2Distance(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        D0, D1 = in_features, out_features
        w: Tensor = get_uniform_tensor(lb=-0.1, ub=0.1, B=D0, Din=D1).detach()
        self.w = nn.Parameter(w)

    # def __repr__(self):
    #     return f'View{self.shape}'

    def forward(self, input: Tensor) -> Tensor:
        """
        Outputs l2 norm of the batch of inputs with the collection of trainable templates:
            forward(X) = Z = ||X - W||^2_2
        such that Z[b, d] = ||x^(b) - w^(l)_d||^2_2.
        Dimensionality transform:
            [B, D0] -> [B, D1]

        ref:
            - https://www.evernote.com/shard/s410/sh/6c127a6b-8faa-ca99-4475-9b9cc91fe8c7/c00b07b7f2f36b62bea317f74ceb7588
        :param input: [B, D0]
        :return:
        """
        B, D0 = input.size()
        D00, D1 = self.w.size()
        assert D0 == D00
        X, W = input, self.w
        # - compute comparison of examples
        x_dot_x: Tensor = (X * X).sum(dim=1, keepdims=True)  # [B, 1] = red([B, D0] * [B, D0], dim=1)
        assert x_dot_x.size() == torch.Size([B, 1])
        # - compute comparison of templates
        w_dot_w: Tensor = (W * W).sum(dim=0, keepdims=True)  # [1, D1] = red([D0, D1] * [D0, D1], dim=0)
        assert w_dot_w.size() == torch.Size([1, D1])
        # - compute comparison of templates with batch examples
        w_dot_w = X @ W  # [B, D1] = [B, D0] * [D0, D1]
        assert w_dot_w.size() == torch.Size([B, D1])
        # - compute vectorized l2 norm for batch of example with collection of templates
        batch_l2_norm: Tensor = x_dot_x - 2 * w_dot_w + w_dot_w
        assert batch_l2_norm.size() == torch.Size([B, D1])
        return batch_l2_norm


class HBF(nn.Module):
    def __init__(self):
        super().__init__()

    # def __repr__(self):
    #     return f'View{self.shape}'

    def forward(self, input):
        pass


# - tests

def norm2_test():
    B, Din = 4, 5
    Dout = 6
    x = torch.randn(B, Din)
    # - lL2Norm layer forward
    l2_layer: L2Distance = L2Distance(in_features=Din, out_features=Dout)
    out: Tensor = l2_layer(x)
    print(out)
    assert out.size() == torch.Size([B, Dout])
    # - equality test
    # out2: Tensor = torch.linalg.norm(x - l2_layer.w.detach(), dim=0)
    # print(out2)
    # assert (out2 == out2).all()


if __name__ == '__main__':
    norm2_test()
    print('Done! success \a.')