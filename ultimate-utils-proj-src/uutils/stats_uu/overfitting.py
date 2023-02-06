"""
Estimates overfitting of a model by comparing the training and test error/loss/accuracy.
Note overfitting is NOT fitting the data perfectly, not is it interpolation. It's a large difference between the train and test metrics.
In a perfect world it's the empirical risk - the expected risk.

ref:
    - https://en.wikipedia.org/wiki/Generalization_error
"""
from typing import Optional

import logging

import numpy as np


def compute_generalization_gap(train_metrics: list[float],
                               test_metrics: list,
                               metric_name: Optional[str] = None,
                               ) -> float:
    """
    Compute the generalization gap between the training and test metrics. This is the difference between the
    train and test metrics e.g. test loss - train loss. In this example the test loss is usually higher than the train
    for sufficiently large N, and if the gap is large then the model might be overfitting. Large is arbitrary, but
    in CVPR they accept apper based on 1-2% so if the gap is less than that then it's probably not overfitting.

    Note:
        - gen gap should be positive for accs
        - gen gap should be negative for losses (i.e. train loss is usually small than eval loss.

    ref:
        - https://en.wikipedia.org/wiki/Generalization_error
    """
    # - compute generalization gap
    train_metric: float = np.mean(train_metrics)
    test_metric: float = np.mean(test_metrics)
    gen_gap: float = test_metric - train_metric
    if metric_name == 'acc':
        # gen gap using acc test acc - train acc should < 0 since test acc should be worse (lower) than train acc for sufficiently large N
        logging.warning(f'Gen gap using acc should be {gen_gap=}')
    elif metric_name == 'loss':
        # gen gap using loss test loss - train loss should > 0 since test loss should be worse (higher) than train loss for sufficiently large N
        logging.warning(f'Gen gap using loss should be {gen_gap=}')
    return gen_gap


def aic(losses: list[float], num_params: int) -> float:
    """
    Compute the Akaike Information Criterion (AIC) for a model. Models with lower AIC are preferred.

    Notes:
        -  AIC estimates the relative amount of information lost by a given model: the less information a model loses,
        the higher the quality of that model. In other words, AIC deals with both the risk of overfitting and the risk of underfitting.

    """
    log_likelihood = -np.sum(losses)
    return -2 * log_likelihood + 2 * num_params


def bic(losses: list[float], num_params: int, num_observations: int) -> float:
    """
    Compute the Bayesian Information Criterion (BIC) for a model. Models with lower BIC are generally preferred.

    Note:
        - When fitting models, it is possible to increase the likelihood by adding parameters, but doing so may result in overfitting. Both BIC and AIC attempt to resolve this problem by introducing a penalty term for the number of parameters in the model; the penalty term is larger in BIC than in AIC for sample sizes greater than 7.
    """
    log_likelihood = -np.sum(losses)
    return -2 * log_likelihood + num_params * np.log(num_observations)


# -- example

def synthetic_aic_bis():
    # Example usage
    losses = [0.1, 0.2, 0.3, 0.4, 0.5]  # replace with a list of cross-entropy losses for your model
    num_params = 20  # replace with the number of parameters in your model
    num_observations = len(losses)  # replace with the number of observations in your data

    aic_value = aic(losses, num_params)
    bic_value = bic(losses, num_params, num_observations)

    print("AIC: ", aic_value)
    print("BIC: ", bic_value)


# - run it

if __name__ == '__main__':
    import time

    start = time.time()
    # - run it
    synthetic_aic_bis()
    # - Done
    print(f'Done in {time.time() - start:.2f} seconds')
