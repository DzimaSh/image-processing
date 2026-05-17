"""Unit tests for fixed-point CDF 9/7 Wavelet Lifting Scheme."""

import numpy as np
import pytest
from src.core.lifting_scheme import (
    fwd_dwt_1d,
    inv_dwt_1d,
    fwd_dwt_2d,
    inv_dwt_2d,
    SCALE_FACTOR,
)
from src.core.metrics import calculate_psnr


def test_dwt_1d_reversibility() -> None:
    """Test 1D DWT and IDWT roundtrip reconstructs the original signal.

    Verifies that the reconstructed signal matches the original signal within
    a very small tolerance (due to fixed-point rounding).
    """
    # Create a simple smooth 1D signal of even length
    orig = np.array([12, 45, 78, 110, 89, 56, 32, 10], dtype=np.int64)

    # Scale to Q16
    orig_q16 = orig << 16

    # Forward
    s, d = fwd_dwt_1d(orig_q16)

    # Inverse
    recon_q16 = inv_dwt_1d(s, d)

    # Convert back to standard representation
    recon = recon_q16.astype(np.float64) / SCALE_FACTOR

    # Assert near-exact matching (error < 1e-4)
    np.testing.assert_allclose(recon, orig, rtol=1e-4, atol=1e-4)


def test_dwt_2d_reversibility_level1() -> None:
    """Test 1-level 2D DWT and IDWT reconstructs a grayscale uint8 image.

    The Peak Signal-to-Noise Ratio (PSNR) should be exceptionally high
    (typically > 80 dB or even infinite) indicating high-fidelity reconstruction.
    """
    # Create a 2D synthetic test image
    np.random.seed(42)
    orig_img = np.random.randint(0, 256, (32, 32)).astype(np.uint8)

    # Forward DWT (Level 1)
    coeffs = fwd_dwt_2d(orig_img, levels=1)

    # Inverse DWT
    recon_img = inv_dwt_2d(coeffs, levels=1)

    # Calculate PSNR between original and reconstructed
    psnr = calculate_psnr(orig_img, recon_img)

    # Ensure PSNR is very high (greater than 60 dB or infinite)
    assert psnr > 60.0 or np.isinf(psnr)


def test_dwt_2d_multi_level() -> None:
    """Test 3-level 2D DWT and IDWT reconstructs correctly."""
    # Create a 2D synthetic test image (size must be divisible by 2^3 = 8)
    np.random.seed(123)
    orig_img = np.random.randint(0, 256, (64, 64)).astype(np.uint8)

    # Forward DWT (Level 3)
    coeffs = fwd_dwt_2d(orig_img, levels=3)

    # Inverse DWT
    recon_img = inv_dwt_2d(coeffs, levels=3)

    # Verify reconstruction is highly accurate
    np.testing.assert_allclose(recon_img, orig_img, rtol=1e-2, atol=2e-2)
    
    psnr = calculate_psnr(orig_img, recon_img)
    assert psnr > 60.0 or np.isinf(psnr)


def test_dwt_1d_odd_length_error() -> None:
    """Test that DWT raises a ValueError on odd-length signals."""
    odd_signal = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="Signal length must be even"):
        fwd_dwt_1d(odd_signal)
