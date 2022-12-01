from pprint import pprint

import numpy as np

from dataclasses import dataclass


@dataclass
class MomentCI:
    mom: float
    ci: float
    moment_idx: int


def _my_compute_central_moment(a: np.ndarray, moment_idx: int = 1, centered: bool = True) -> float:
    """
    Computes central moment: mom_n = E_x[(X - mu)^n], note, first central moment is always zero.
    """
    if centered:
        a: np.ndarray = a - a.mean()
    a_n: np.ndarray = a ** moment_idx
    mom: float = float(a_n.mean())
    return mom


def compute_uncentered_moments(a: np.ndarray, moment_idx: int = 1) -> float:
    """"""
    mom: float = _my_compute_central_moment(a, moment_idx, centered=False)
    return mom


def get_diagonal(matrix: np.ndarray,
                 check_if_symmetric: bool = False,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get's the diagonal from a matrix.

    ref: remove diagonal, https://stackoverflow.com/questions/46736258/deleting-diagonal-elements-of-a-numpy-array
    """
    triu: np.ndarray = np.triu(matrix)
    tril: np.ndarray = np.tril(matrix)
    # distance_matrix = distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool)].reshape(distance_matrix.shape[0], -1)
    # remove diagonal and dummy zeros where the other triangular matrix was artificially placed.
    distance_matrix: np.ndarray = triu[triu != 0.0]

    # - check we are returning diagonal so sit's samller than full matrix
    size_full_matrix: int = matrix.size
    size_diag: int = distance_matrix.size
    assert size_diag < size_full_matrix, f'The diagonal matrix is not being extracted correct:' \
                                         f'\n{size_diag=}, {size_full_matrix=}'

    # - flatten
    flatten: np.ndarray = distance_matrix.flatten()
    # - check is of size (N,)
    assert flatten.shape == (flatten.shape[0],)
    assert len(flatten.shape) == 1
    assert isinstance(flatten.shape[0], int)
    if check_if_symmetric:
        from uutils.torch_uu import approx_equal
        assert approx_equal(triu.sum(), tril.sum(),
                            tolerance=1e-4), f'Distance matrix is not symmetric, are you sure this is correct?'
        assert approx_equal(distance_matrix.mean(), triu[triu != 0.0].mean(),
                            tolerance=1e-4), f'Mean should be equal to triangular matrix'
        # assert approx_equal(mu, triu[triu != 0.0].mean(), tolerance=1e-4)
    return flatten, triu, tril


def compute_moments(array: np.ndarray,
                    moment_idxs: list[int] = [1],
                    confidence: float = 0.95,
                    centered: bool = True,
                    ) -> dict[dict]:
    """
    Moment:
    0 = total
    1 = mean
    2 = variance
    3 = skewness
    4 = kurtosis
    ...
    """
    from uutils.torch_uu.metrics.confidence_intervals import nth_central_moment_and_its_confidence_interval
    # - check is of size (N,)
    assert array.shape == (array.shape[0],)
    assert len(array.shape) == 1
    assert isinstance(array.shape[0], int)
    # - compute moments
    moments: dict[dict] = {}
    moment_idx: int
    for moment_idx in moment_idxs:
        mom, ci = nth_central_moment_and_its_confidence_interval(array, moment_idx, confidence, centered)
        moments[moment_idx] = MomentCI(mom=mom, ci=ci, moment_idx=moment_idx)
    return moments


def standardized_moments():
    """ mom / std^n"""
    pass


# - test

def compute_moments_test():
    n: int = 500
    a: np.ndarray = np.random.normal(loc=2.0, scale=8.0, size=n)
    moments: dict = compute_moments(a, moment_idxs=[1, 2, 3, 4])
    pprint(moments)
    moments: dict = compute_moments(a, moment_idxs=[1, 2, 3, 4], centered=False)
    pprint(moments)


if __name__ == '__main__':
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    compute_moments_test()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
