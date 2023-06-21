"""
Chinchilla scaling laws aim to compute optimal model size and number of tokens to train on for a given compute budget.
In particular the optimization equation form in their paper is:

    N_opt(C), D_opt(C) = argmin_{N, D s.t. FLOPs(N,D) = C} L(N, D)

Where N is the number of model parameters,
D is the number of training tokens,
C=FLOPs(N, D, its) where it's can be ignored since it's the number of tokens D.
FLOPS = floating point operations per second (FLOPS, flops or flop/s).


Note:
    - Chinchilla 70B=70e109 was trained on 1.4T=1.4e12 tokens (stated for intuition).

ref:
    - Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
"""


def chinchilla_scaling_law_estimate_num_params_num_tokens_gpt2_sudharsan_sundar(desired_loss: float,
                                                                                compute_budget_flps: float,
                                                                                its: int = None,
                                                                                # for completeness, but it should be same a number of tokens, since Chinchilla goes through data set once
                                                                                ) -> tuple[float, float]:
    """Get number of parameters N and number of tokens D to train on for a given compute budget C=FLOPs(N,D)=FLOPs(N,D,its=D)."""
    num_params: float = -1
    num_tokens: float = -1
    # TODO
    return num_params, num_tokens


def num_params_num_tokens_chinchilla_approach1_fix_mdl_size_vary_train_tokens() -> int:
    pass


def num_params_num_tokens_chinchilla_appraoch2_isoflop_profiles() -> int:
    pass


def num_params_num_tokens_chinchilla_approach3_fitting_a_parametric_loss() -> int:
    pass


def num_params_num_tokens_chinchilla_approach3_fitting_a_parametric_loss() -> int:
    """
    L^(N, D) = E + A * N^-alpha + B * D^-beta

    estimate A, B, alpha, beta using huber loss with L_BGFS. Delta=1e-3.

    TODO: 3.3 has alpha = 0.46 b=0.54...does this matter and other equations....
    """
    pass
