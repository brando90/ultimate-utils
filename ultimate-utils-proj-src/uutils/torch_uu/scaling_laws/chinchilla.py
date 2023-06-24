"""
Chinchilla scaling laws aim to compute optimal model size and number of tokens to train on for a given compute budget.
In particular the optimization equation form in their paper is:

    N_opt(C), D_opt(C) = argmin_{N, D s.t. FLOPs(N,D) = C} L(N, D)

Where N is the number of model parameters,
D is the number of training tokens,
C=FLOPs(N, D, its) where its can be ignored since it's the number of tokens D.
FLOPS = floating point operations per second (FLOPS, flops or flop/s).


Note:
    - Chinchilla 70B=7e10 was trained on 1.4T=1.4e12 tokens (stated for intuition).

ref:
    - Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
"""
from typing import Tuple
import numpy as np


def chinchilla_scaling_law_estimate_num_params_num_tokens_gpt2_sudharsan_sundar(compute_budget_flps: float,
                                                                                desired_loss: float = None,
                                                                                its: int = None,
                                                                                # for completeness, but it should be
                                                                                # same a number of tokens, since
                                                                                # Chinchilla goes through data set once
                                                                                ) -> Tuple[float, float]:
    """Get number of parameters N and number of tokens D to train on for a given compute budget
    C=FLOPs(N,D)=FLOPs(N,D,its=D)."""

    """
    Note: Chinchilla paper describes 3 approaches, which are all similar but have meaningful differences at very large 
    scales (e.g. 20B+ params). I use approach #1 for this function, as it is most prominently featured in the paper.
    
    High level principles:
        - Parameter count and tokens should be scaled equally. E.g. if you want a 10x larger model, you should then 
        train on 10x num tokens
        
        - According to Kaplan et. al. (2020) (which Chinchilla paper uses), FLOPs(N, D) = 6 * N * D
        
        - Rough approximation: params:tokens is approx. 1:20
            - Highly accurate in the range of ~400M - ~10B. Accuracy begins to deteriorate at larger scales (100B+)
            - This is basically the technique I used to estimate the number of tokens to train on for my experiments
        
        - More precise approach below
        
    Example with different approach: https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb
    """

    # Rough calculation
    num_params: float = (compute_budget_flps / (6 * 20)) ** 0.5
    num_tokens: float = ((compute_budget_flps * 20) / 6) ** 0.5

    return num_params, num_tokens


def num_params_num_tokens_chinchilla_approach1_fix_mdl_size_vary_train_tokens(compute_budget_flps: float) -> Tuple[float, float]:
    """
    Precise calculation of Approach #1:
        - I essentially use the lines from Figure 1 in the Chinchilla paper.
        - I find these lines by fitting a linear function to the optimal values described in Table 3
            - I log both compute and params by 10 to make the equation easier to calculate/linear

    """

    logged_compute = np.log10(compute_budget_flps)
    logged_params = ((8.603 - 13.0) / (19.283 - 28.114)) * logged_compute + (8.602 - 9.603)
    logged_tokens = ((0.903 - 5.335) / (19.283 - 28.114)) * logged_compute + (0.903 - 9.678)

    num_params: float = np.power(10, logged_params)
    num_tokens: float = np.power(10, logged_tokens) * np.power(10, 9)
    # NOTE: I can make a more general implementation/fill in the other approaches, but it doesn't seem like it's
    # central to what we're trying to do right now.

    return num_params, num_tokens


def num_params_num_tokens_chinchilla_appraoch2_isoflop_profiles() -> int:
    """
    Optimal param count is proportional to compute^{0.49}
    Optimal token count is proportional to compute^{0.51}

    Table A3, pg 26
    """

    pass


def num_params_num_tokens_chinchilla_approach3_fitting_a_parametric_loss() -> int:
    """
    Optimal param count is proportional to compute^{0.46}
    Optimal token count is proportional to compute^{0.54}

    Table A3, pg 26
    """

    pass


def num_params_num_tokens_chinchilla_approach3_fitting_a_parametric_loss_v2() -> int:
    """
    L^(N, D) = E + A * N^-alpha + B * D^-beta

    estimate A, B, alpha, beta using huber loss with L_BGFS. Delta=1e-3.

    TODO: 3.3 has alpha = 0.46 b=0.54...does this matter and other equations....
    """

    """
    My interpretation/understanding: fitting the L^ equation to minimums of the Huber losses gets you alpha=x, and 
    beta=y for this approach. So, we can then back out a=.46, b=.54 from this to find out the relationship between 
    compute scaling, and param and data scaling. It seems like this final form (using a and b) is what the paper 
    actually uses as its approach, and not the intermediate step involving alpha and beta, so using a and b seem
    like the way the paper says we should consider scaling laws.
    
    That being said, it's not clear to me whether the form using alpha and beta is exactly the same as using a and b, 
    or if the form using a and b is only a close approximation.  
    """

    pass


# Sanity check

# Compute for 1B model
compute_1 = 1.21 * 10**20

# Compute for 67B model
compute_2 = 5.76 * 10**23

# Half the compute for 280B model
compute_3 = 9.9 * 10**24

print('Rough estimate (params, tokens): ',
      chinchilla_scaling_law_estimate_num_params_num_tokens_gpt2_sudharsan_sundar(compute_1)[0] / 10**9,
      chinchilla_scaling_law_estimate_num_params_num_tokens_gpt2_sudharsan_sundar(compute_1)[1] / 10**9,
      'Actual (params, tokens): ',
      1000000000 / 10**9,
      20200000000 / 10**9)
print('Rough estimate (params, tokens): ',
      chinchilla_scaling_law_estimate_num_params_num_tokens_gpt2_sudharsan_sundar(compute_2)[0] / 10**9,
      chinchilla_scaling_law_estimate_num_params_num_tokens_gpt2_sudharsan_sundar(compute_2)[1] / 10**9,
      'Actual (params, tokens): ',
      67000000000 / 10**9,
      1500000000000 / 10**9)
print('Rough estimate (params, tokens): ',
      chinchilla_scaling_law_estimate_num_params_num_tokens_gpt2_sudharsan_sundar(compute_3)[0] / 10**9,
      chinchilla_scaling_law_estimate_num_params_num_tokens_gpt2_sudharsan_sundar(compute_3)[1] / 10**9,
      'Actual (params, tokens): ',
      280000000000 / 10**9,
      5900000000000 / 10**9)

print('Precise estimate (params, tokens): ',
      num_params_num_tokens_chinchilla_approach1_fix_mdl_size_vary_train_tokens(compute_1)[0] / 10**9,
      num_params_num_tokens_chinchilla_approach1_fix_mdl_size_vary_train_tokens(compute_1)[1] / 10**9,
      'Actual (params, tokens): ',
      1000000000 / 10**9,
      20200000000 / 10**9)
print('Precise estimate (params, tokens): ',
      num_params_num_tokens_chinchilla_approach1_fix_mdl_size_vary_train_tokens(compute_2)[0] / 10**9,
      num_params_num_tokens_chinchilla_approach1_fix_mdl_size_vary_train_tokens(compute_2)[1] / 10**9,
      'Actual (params, tokens): ',
      67000000000 / 10**9,
      1500000000000 / 10**9)
print('Precise estimate (params, tokens): ',
      num_params_num_tokens_chinchilla_approach1_fix_mdl_size_vary_train_tokens(compute_3)[0] / 10**9,
      num_params_num_tokens_chinchilla_approach1_fix_mdl_size_vary_train_tokens(compute_3)[1] / 10**9,
      'Actual (params, tokens): ',
      280000000000 / 10**9,
      5900000000000 / 10**9)

