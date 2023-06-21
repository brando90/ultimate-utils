"""
Collection of FLOPs estimate.

Thoughts:
Idk if this really matters to us in academia. I don't pay per FLOP. I can just run my job for as long as the SNAP cluster
users allow me -- without complaining.
My thought is that it's more important to decide for a fair comparison of methods:
number of token Ds, batch size B, model size N, optimizer, (etc.) and fix the number iterations its.
Say its=D then or some multiple of D for fine-tuning.

ref:
    - what language model to train if you have one million gpu hours? https://arxiv.org/abs/2210.15424
"""
def flops_according_to_big_science_architecture_and_scaling_group_and_kaplan(N: float, D: float) -> int:
    """
    C=6ND.
    Read 2.2 more carefully.

    ref:
        - what language model to train if you have one million gpu hours? https://arxiv.org/abs/2210.15424
    """
    C = 6 * N * D
    return C


def flops_according_to_kaplan_forward_pass(n_layer, n_ctx, d) -> int:
    """
    ref:
        - what language model to train if you have one million gpu hours? https://arxiv.org/abs/2210.15424
    """
    C_forward: int = 2 * (12 * n_layer * d**2 + n_layer * n_ctx * d)
    return C_forward


def flops_estimate_according_to_chichilla() -> int:
    """
    ref:
        - Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
    """
    pass