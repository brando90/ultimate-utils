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
from matplotlib import pyplot as plt


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

    # 'Hand-computed' precise solution
    # logged_compute = np.log10(compute_budget_flps)
    # logged_params = ((8.603 - 13.0) / (19.283 - 28.114)) * logged_compute + (8.602 - 9.603)
    # logged_tokens = ((0.903 - 5.335) / (19.283 - 28.114)) * logged_compute + (0.903 - 9.678)
    #
    # num_params: float = np.power(10, logged_params)
    # num_tokens: float = np.power(10, logged_tokens) * np.power(10, 9)

    return num_params, num_tokens


def num_params_num_tokens_chinchilla_approach1_fix_mdl_size_vary_train_tokens(compute_budget_flps: float) -> Tuple[float, float]:
    """
    Precise calculation of Approach #1:
        - I essentially use the lines from Figure 1 in the Chinchilla paper.
        - I find these lines by fitting a linear function to the optimal values described in Table 3
            - I log both compute and params by 10 to make the equation easier to calculate/linear

    """

    # Code adapted from https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb
    raw = [
        [1.92e19, 400e6, 8e9],
        [1.21e20, 1e9, 20.2e9],
        [1.23e22, 10e9, 205.1e9],
        [5.76e23, 67e9, 1.5e12],
        [3.85e24, 175e9, 3.7e12],
        [9.9e24, 280e9, 5.9e12],
        [3.43e25, 520e9, 11e12],
        [1.27e26, 1e12, 21.2e12],
        [1.3e28, 10e12, 216.2e12],
    ]

    x = np.array([np.log10(x[0]) for x in raw])
    y = np.array([np.log10(x[1]) for x in raw])
    y2 = np.array([np.log10(x[2]) for x in raw])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    m2, c2 = np.linalg.lstsq(A, y2, rcond=None)[0]
    # print(f"y = {m}x + {c}")
    # print(f"y2 = {m2}x + {c2}")

    logged_compute = np.log10(compute_budget_flps)
    logged_params = m * logged_compute + c
    logged_tokens = m2 * logged_compute + c2

    num_params: float = np.power(10, logged_params)
    num_tokens: float = np.power(10, logged_tokens)

    return num_params, num_tokens


def num_params_num_tokens_chinchilla_appraoch2_isoflop_profiles(compute_budget_flps: float) -> Tuple[float, float]:
    """
    Optimal param count is proportional to compute^{0.49}
    Optimal token count is proportional to compute^{0.51}
    
    Precise calculation of Approach #2:
        - I essentially use the lines from Figure 1 in the Chinchilla paper.
        - I find these lines by fitting a linear function to the optimal values described in Table A3 (pg 26)
            - I log both compute and params by 10 to make the equation easier to calculate/linear
    """

    # Code adapted from https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb
    raw = [
        [1.84e19, 400e6, 7.7e9],
        [1.20e20, 1e9, 20.0e9],
        [1.32e22, 10e9, 219.5e9],
        [6.88e23, 67e9, 1.7e12],
        [4.54e24, 175e9, 4.3e12],
        [1.18e25, 280e9, 7.1e12],
        [4.19e25, 520e9, 13.4e12],
        [1.59e26, 1e12, 26.5e12],
        [1.75e28, 10e12, 292.0e12],
    ]

    x = np.array([np.log10(x[0]) for x in raw])
    y = np.array([np.log10(x[1]) for x in raw])
    y2 = np.array([np.log10(x[2]) for x in raw])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    m2, c2 = np.linalg.lstsq(A, y2, rcond=None)[0]
    # print(f"y = {m}x + {c}")
    # print(f"y2 = {m2}x + {c2}")

    logged_compute = np.log10(compute_budget_flps)
    logged_params = m * logged_compute + c
    logged_tokens = m2 * logged_compute + c2

    num_params: float = np.power(10, logged_params)
    num_tokens: float = np.power(10, logged_tokens)

    return num_params, num_tokens


def num_params_num_tokens_chinchilla_approach3_fitting_a_parametric_loss(compute_budget_flps: float) -> Tuple[float, float]:
    """
    Optimal param count is proportional to compute^{0.46}
    Optimal token count is proportional to compute^{0.54}

    Precise calculation of Approach #3:
        - I essentially use the lines from Figure 1 in the Chinchilla paper.
        - I find these lines by fitting a linear function to the optimal values described in Table A3 (pg 26)
            - I log both compute and params by 10 to make the equation easier to calculate/linear
    """

    # Code adapted from https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb
    # Found error in Table A3: row 5 has compute 1 OOM lower than it should be
    raw = [
        [2.21e19, 400e6, 9.2e9],
        [1.62e20, 1e9, 27.1e9],
        [2.46e22, 10e9, 410.1e9],
        [1.71e24, 67e9, 4.1e12],
        [1.26e25, 175e9, 12.0e12],
        [3.52e25, 280e9, 20.1e12],
        [1.36e26, 520e9, 43.5e12],
        [5.65e26, 1e12, 94.1e12],
        [8.55e28, 10e12, 1425.5e12],
    ]

    x = np.array([np.log10(x[0]) for x in raw])
    y = np.array([np.log10(x[1]) for x in raw])
    y2 = np.array([np.log10(x[2]) for x in raw])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    m2, c2 = np.linalg.lstsq(A, y2, rcond=None)[0]
    # print(f"y = {m}x + {c}")
    # print(f"y2 = {m2}x + {c2}")

    x_hat = np.linspace(20, 28, 100)
    y_hat = m * x_hat + c

    plt.plot(x, y)
    plt.plot(x_hat, y_hat)
    plt.show()

    logged_compute = np.log10(compute_budget_flps)
    logged_params = m * logged_compute + c
    logged_tokens = m2 * logged_compute + c2

    num_params: float = np.power(10, logged_params)
    num_tokens: float = np.power(10, logged_tokens)

    return num_params, num_tokens


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
    like the way the paper says we should consider scaling laws. (and spending extra time/effort trying to figure out
    the difference doesn't seem especially useful.)
    
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

print('Approach 2 estimate (params, tokens): ',
      num_params_num_tokens_chinchilla_appraoch2_isoflop_profiles(1.84e19)[0] / 10**9,
      num_params_num_tokens_chinchilla_appraoch2_isoflop_profiles(1.84e19)[1] / 10**9,
      'Actual (params, tokens): ',
      .4,
      7.7)
print('Approach 2 estimate (params, tokens): ',
      num_params_num_tokens_chinchilla_appraoch2_isoflop_profiles(6.88e23)[0] / 10**9,
      num_params_num_tokens_chinchilla_appraoch2_isoflop_profiles(6.88e23)[1] / 10**9,
      'Actual (params, tokens): ',
      67,
      1700)
print('Approach 2 estimate (params, tokens): ',
      num_params_num_tokens_chinchilla_appraoch2_isoflop_profiles(4.19e25)[0] / 10**9,
      num_params_num_tokens_chinchilla_appraoch2_isoflop_profiles(4.19e25)[1] / 10**9,
      'Actual (params, tokens): ',
      520,
      13400)

print('Approach 3 estimate (params, tokens): ',
      num_params_num_tokens_chinchilla_approach3_fitting_a_parametric_loss(2.21e19)[0] / 10**9,
      num_params_num_tokens_chinchilla_approach3_fitting_a_parametric_loss(2.21e19)[1] / 10**9,
      'Actual (params, tokens): ',
      .4,
      9.2)
print('Approach 3 estimate (params, tokens): ',
      num_params_num_tokens_chinchilla_approach3_fitting_a_parametric_loss(1.71e24)[0] / 10**9,
      num_params_num_tokens_chinchilla_approach3_fitting_a_parametric_loss(1.71e24)[1] / 10**9,
      'Actual (params, tokens): ',
      67,
      4100)
print('Approach 3 estimate (params, tokens): ',
      num_params_num_tokens_chinchilla_approach3_fitting_a_parametric_loss(1.36e26)[0] / 10**9,
      num_params_num_tokens_chinchilla_approach3_fitting_a_parametric_loss(1.36e26)[1] / 10**9,
      'Actual (params, tokens): ',
      520,
      43500)

