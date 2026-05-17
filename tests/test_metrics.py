"""Unit tests for statistical quality metrics (MSE and PSNR)."""

import numpy as np
import pytest
from src.shared.metrics import calculate_mse, calculate_psnr


def test_calculate_mse_identical() -> None:
    """Test that MSE is exactly zero for identical images."""
    img1 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    img2 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    assert calculate_mse(img1, img2) == 0.0


def test_calculate_mse_basic() -> None:
    """Test standard MSE calculations."""
    img1 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    img2 = np.array([[12, 18], [28, 41]], dtype=np.uint8)
    # Differences: [2, -2, -2, 1]
    # Squared differences: [4, 4, 4, 1]
    # Sum: 13, Mean: 13/4 = 3.25
    assert calculate_mse(img1, img2) == pytest.approx(3.25)


def test_calculate_mse_shape_mismatch() -> None:
    """Test that a shape mismatch raises a ValueError."""
    img1 = np.zeros((3, 3))
    img2 = np.zeros((3, 4))
    with pytest.raises(ValueError, match="shapes do not match"):
        calculate_mse(img1, img2)


def test_calculate_psnr_identical() -> None:
    """Test that identical images return infinity for PSNR."""
    img1 = np.zeros((5, 5))
    img2 = np.zeros((5, 5))
    assert calculate_psnr(img1, img2) == float("inf")


def test_calculate_psnr_standard() -> None:
    """Test standard PSNR calculation against an analytical value."""
    img1 = np.zeros((10, 10))
    img2 = np.ones((10, 10)) * 5.0
    # MSE is 25.0
    # PSNR = 10 * log10(255^2 / 25) = 10 * log10(65025 / 25) = 10 * log10(2601)
    # PSNR ~ 34.1514 dB
    expected = 10.0 * np.log10(255.0 ** 2 / 25.0)
    assert calculate_psnr(img1, img2) == pytest.approx(expected)
