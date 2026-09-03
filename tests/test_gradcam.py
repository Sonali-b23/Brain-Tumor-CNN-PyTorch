import numpy as np
import torch

from src.gradcam import IMAGENET_MEAN, IMAGENET_STD, unnormalize


def test_unnormalize_uses_correct_per_channel_stats():
    """Regression test for the original bug: the old code used the red
    channel's mean/std (0.485, 0.229) as a scalar for all three channels,
    which is only correct by coincidence for red and silently wrong for
    green and blue."""
    torch.manual_seed(0)
    normalized = torch.rand(3, 4, 4)  # CHW, values in [0, 1) pre-unnormalize

    result = unnormalize(normalized)

    # Recompute the expected result by hand with full per-channel arrays
    img = normalized.permute(1, 2, 0).numpy()
    expected = img * IMAGENET_STD + IMAGENET_MEAN
    expected = np.clip(expected, 0, 1)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_unnormalize_would_differ_from_buggy_scalar_version():
    """Sanity check that the fix actually changes behavior versus the old
    scalar-based formula (i.e. this test would have failed against the
    original buggy code)."""
    torch.manual_seed(1)
    normalized = torch.rand(3, 4, 4)

    correct = unnormalize(normalized)

    img = normalized.permute(1, 2, 0).numpy()
    buggy = img * 0.229 + 0.485
    buggy = np.clip(buggy, 0, 1)

    # Green and blue channels should disagree with the buggy scalar formula
    assert not np.allclose(correct[:, :, 1], buggy[:, :, 1])
    assert not np.allclose(correct[:, :, 2], buggy[:, :, 2])


def test_unnormalize_output_is_clipped_to_valid_range():
    torch.manual_seed(2)
    normalized = torch.rand(3, 8, 8) * 3 - 1.5  # push values outside [0,1) after unnormalizing
    result = unnormalize(normalized)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
