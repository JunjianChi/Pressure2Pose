import numpy as np
import torch

from crosstalk.metrics import MaskedScore, masked_mse


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    target = rng.random((4, 1, 6, 5)).astype(np.float32)
    mask = (rng.random((4, 1, 6, 5)) > 0.3).astype(np.float32)
    return target, mask


def test_perfect_prediction_scores_zero_error_and_unit_r2():
    target, mask = _batch()
    score = MaskedScore()
    score.update(target, target, mask)
    assert score.mse == 0.0
    assert score.r2 == 1.0


def test_mean_prediction_scores_zero_r2():
    target, mask = _batch()
    mean = (mask * target).sum() / mask.sum()
    score = MaskedScore()
    score.update(np.full_like(target, mean), target, mask)
    assert abs(score.r2) < 1e-6


def test_masked_cells_never_enter_the_score():
    target, mask = _batch()
    pred = target.copy()
    corrupted = target.copy()
    corrupted[mask == 0.0] += 100.0
    score = MaskedScore()
    score.update(pred, corrupted, mask)
    assert score.mse == 0.0


def test_streaming_equals_whole_array_and_loss():
    target, mask = _batch()
    pred = target + 0.1 * np.float32(1.0)
    whole = MaskedScore()
    whole.update(pred, target, mask)
    streamed = MaskedScore()
    for i in range(len(target)):
        streamed.update(pred[i], target[i], mask[i])
    assert np.isclose(streamed.mse, whole.mse)
    assert np.isclose(streamed.r2, whole.r2)
    loss = masked_mse(torch.from_numpy(pred), torch.from_numpy(target), torch.from_numpy(mask))
    assert np.isclose(float(loss), whole.mse)
