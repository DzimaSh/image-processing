"""Metrics module for analyzing reconstructed image quality.

This module provides functions to calculate Mean Squared Error (MSE) and
Peak Signal-to-Noise Ratio (PSNR) to evaluate reconstruction accuracy.
"""

import numpy as np


def calculate_mse(orig: np.ndarray, recon: np.ndarray) -> float:
    """Calculate the Mean Squared Error between two images.

    Args:
        orig: The original image as a NumPy array (grayscale, float or uint8).
        recon: The reconstructed image as a NumPy array of same shape.

    Returns:
        The calculated MSE value.

    Raises:
        ValueError: If input arrays have mismatching shapes.
    """
    if orig.shape != recon.shape:
        raise ValueError(
            f"Image shapes do not match: {orig.shape} vs {recon.shape}"
        )

    # Convert to float64 to prevent overflow during subtraction
    diff = orig.astype(np.float64) - recon.astype(np.float64)
    mse = np.mean(diff ** 2)
    return float(mse)


def calculate_psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Assumes a peak dynamic range of 255.0 (for 8-bit images).

    Args:
        orig: The original image as a NumPy array.
        recon: The reconstructed image as a NumPy array.

    Returns:
        The PSNR value in dB, or float('inf') if MSE is exactly 0 (identical).
    """
    mse = calculate_mse(orig, recon)
    if mse == 0.0:
        return float("inf")

    max_val = 255.0
    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return float(psnr)
