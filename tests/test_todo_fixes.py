"""Tests for the three TODO/FIXME bug fixes in ultimate-utils."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def test_fix1_data_eval_utils_raises_not_implemented_error():
    pytest.importorskip("tenacity")
    from uutils.evals.data_eval_utils import get_iter_for_eval_data_set

    # Calling with a path containing Putnam_MATH_variation_static2 must raise NotImplementedError (not TypeError)
    path_variation = "dummy/path/Putnam_MATH_variation_static2/data"
    with pytest.raises(NotImplementedError) as exc_info:
        get_iter_for_eval_data_set(path_variation)
    assert "Putnam_MATH_variation_static2" in str(exc_info.value)
    assert "not supported yet" in str(exc_info.value)

    # Also verify Putnam-AXIOM raises NotImplementedError
    path_axiom = "dummy/path/Putnam-AXIOM/putnam-axiom-dataset/data"
    with pytest.raises(NotImplementedError) as exc_info_axiom:
        get_iter_for_eval_data_set(path_axiom)
    assert "Putnam-AXIOM" in str(exc_info_axiom.value)
    assert "not supported yet" in str(exc_info_axiom.value)


def test_fix2_collect_hist_arbitrary_classes(monkeypatch):
    # mit_trainer_code has legacy top-level imports that are not used by collect_hist
    for mod in ["data_utils", "utils", "maps", "nn_models"]:
        if mod not in sys.modules:
            monkeypatch.setitem(sys.modules, mod, MagicMock())

    monkeypatch.delitem(sys.modules, "uutils.torch_uu.mit_trainer_code", raising=False)
    from uutils.torch_uu.mit_trainer_code import collect_hist

    torch.manual_seed(42)
    X = torch.randn(7, 4)
    y = torch.zeros(7)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    net = torch.nn.Linear(4, 3)
    device = torch.device("cpu")

    hist = collect_hist(net, dataloader, device)

    # Must give hist.shape == (7, 3), float dtype, and match net(X) row by row
    assert hist.shape == (7, 3)
    assert issubclass(hist.dtype.type, np.floating)
    with torch.no_grad():
        expected = net(X).numpy()
    assert np.allclose(hist, expected)

    # Behaviour for 10 classes must be unchanged
    net10 = torch.nn.Linear(4, 10)
    hist10 = collect_hist(net10, dataloader, device)
    assert hist10.shape == (7, 10)
    with torch.no_grad():
        expected10 = net10(X).numpy()
    assert np.allclose(hist10, expected10)


def test_fix3_task_similarity_normalized_embeddings():
    from uutils.torch_uu.metrics.diversity.task2vec_based_metrics.task_similarity import (
        get_normalized_embeddings,
        get_variance,
    )

    e0 = types.SimpleNamespace(
        hessian=np.array([1.0, 2.0, 3.0]),
        scale=np.array([1.0, 1.0, 1.0]),
    )
    e1 = types.SimpleNamespace(
        hessian=np.array([4.0, 5.0, 6.0]),
        scale=np.array([1.0, 1.0, 1.0]),
    )
    e2 = types.SimpleNamespace(
        hessian=np.array([7.0, 8.0, 9.0]),
        scale=np.array([1.0, 1.0, 1.0]),
    )

    # With embeddings [e0, None, e2], the normalization equals the one computed from [e0, e2] alone
    embeddings_with_none = [e0, None, e2]
    F_with_none, norm_with_none = get_normalized_embeddings(embeddings_with_none)

    F_valid_only, norm_valid_only = get_normalized_embeddings([e0, e2])

    assert np.allclose(norm_with_none, norm_valid_only)
    assert F_with_none.shape == (3, 3)
    # Row 1 of F is all zeros
    assert np.allclose(F_with_none[1], np.zeros(3))
    # Rows 0 and 2 match the valid embeddings' normalized values
    assert np.allclose(F_with_none[0], F_valid_only[0])
    assert np.allclose(F_with_none[2], F_valid_only[1])

    # With no Nones, the output matches the old formula
    embeddings_no_none = [e0, e1, e2]
    F_no_none, norm_no_none = get_normalized_embeddings(embeddings_no_none)

    F_old = np.array([1.0 / get_variance(e, normalized=False) for e in embeddings_no_none])
    norm_old = np.sqrt((F_old ** 2).mean(axis=0, keepdims=True))
    expected_F = F_old / norm_old

    assert np.allclose(norm_no_none, norm_old)
    assert np.allclose(F_no_none, expected_F)
