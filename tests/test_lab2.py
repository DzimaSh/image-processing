"""Unit tests for Lab 2: 2D Non-Separable Wavelet Transforms."""

import numpy as np
import pytest
from src.lab2 import (
    fwd_quaternionic_lifting_2d,
    inv_quaternionic_lifting_2d,
    fwd_non_separable_cdf97_2d,
    inv_non_separable_cdf97_2d,
)
from src.shared.metrics import calculate_psnr, calculate_mse


def test_quaternionic_lifting_reversibility() -> None:
    """Test that Quaternionic block-lifting reconstructs the original image perfectly.

    Since the block-lifting factorization maps integers to integers using rounding,
    the transform is structurally lossless and reversible with zero MSE and infinite PSNR.
    """
    np.random.seed(42)
    # Create a random grayscale image of even dimensions
    orig_img = np.random.randint(0, 256, (64, 64)).astype(np.uint8)

    # Apply forward transform
    coeffs = fwd_quaternionic_lifting_2d(orig_img)

    # Apply inverse transform
    recon_img = inv_quaternionic_lifting_2d(coeffs)

    # Calculate MSE and PSNR
    mse = calculate_mse(orig_img, recon_img)
    psnr = calculate_psnr(orig_img, recon_img)

    # Reconstructed image should be a perfect match
    np.testing.assert_allclose(recon_img, orig_img, rtol=1e-7, atol=1e-7)
    assert mse == 0.0
    assert np.isinf(psnr) or psnr > 100.0


def test_non_separable_cdf97_reversibility() -> None:
    """Test that the 2D Non-Separable CDF 9/7 transform reconstructs highly accurately.

    The Peak Signal-to-Noise Ratio (PSNR) should be exceptionally high (> 60 dB or infinite).
    """
    np.random.seed(123)
    orig_img = np.random.randint(0, 256, (32, 32)).astype(np.uint8)

    # Forward
    coeffs = fwd_non_separable_cdf97_2d(orig_img)

    # Inverse
    recon_img = inv_non_separable_cdf97_2d(coeffs)

    # Check PSNR is extremely high
    psnr = calculate_psnr(orig_img, recon_img)
    assert psnr > 60.0 or np.isinf(psnr)
    np.testing.assert_allclose(recon_img, orig_img, rtol=1e-2, atol=1e-2)


def test_invalid_dimensions() -> None:
    """Test that transforms raise assertion errors for odd-dimension inputs."""
    odd_img = np.zeros((31, 32))
    
    with pytest.raises(AssertionError):
        fwd_quaternionic_lifting_2d(odd_img)
        
    with pytest.raises(AssertionError):
        fwd_non_separable_cdf97_2d(odd_img)
