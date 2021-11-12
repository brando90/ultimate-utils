import numpy as np

from uutils.torch_uu.metrics.cca.cca_core import get_cca_similarity


def svcca_with_keeping_fixed_dims(x: np.ndarray, y: np.ndarray, dims_to_keep: int,
                                  epsilon: float = 1e-10, verbose: bool = False,
                                  full_matrices: bool = False, keepdims: bool = True, axis: int = 1):
    """
    Computes CCA statistics after doing the SV step from SVCCA.
    Incoming data is of shape [D1, N], [D2, N].
    To get svcca similirty do see note.

    Alg:
      - preprocessing:
        - 1) center the incoming raw data
        - 2) SV of centered incoming raw data
        - 3) then cca_core (which does not center but divides by max of incoming data, which due to 2 is centered.)

    Note:
      - To compute svcca distance do: svcca: float = np.mean(svcca_baseline["cca_coef1"])
      - Input data is assumed to be of size [D, N] to make it consistent with the original tutorial: https://github.com/google/svcca/blob/master/tutorials/001_Introduction.ipynb
    """
    # Mean subtract baseline activations
    cx = center(x, axis=axis, keepdims=keepdims)
    cy = center(y, axis=axis, keepdims=keepdims)

    # Perform SVD
    Ux, sx, Vx = np.linalg.svd(cx, full_matrices=full_matrices)
    Uy, sy, Vy = np.linalg.svd(cy, full_matrices=full_matrices)

    svx = np.dot(sx[:dims_to_keep] * np.eye(dims_to_keep), Vx[:dims_to_keep])
    svy = np.dot(sy[:dims_to_keep] * np.eye(dims_to_keep), Vy[:dims_to_keep])

    # Recenter after SVD since CCA assumes incoming stuff is centered
    svx = center(svx, axis=axis, keepdims=keepdims)
    svy = center(svy, axis=axis, keepdims=keepdims)

    svcca_baseline = get_cca_similarity(svx, svy, epsilon=epsilon, verbose=verbose)
    # print("Baseline", np.mean(svcca_baseline["cca_coef1"]), "and MNIST", np.mean(svcca_results["cca_coef1"]))
    svcca: float = np.mean(svcca_baseline["cca_coef1"])
    if verbose:
        print("SVCCA:", svcca)
    return svcca_baseline


def center(x: np.ndarray, axis: int = 1, keepdims=True):
    """
    Centers data assuming data is of shape [D, N].
    """
    cx = x - np.mean(x, axis=axis, keepdims=keepdims)
    return cx
