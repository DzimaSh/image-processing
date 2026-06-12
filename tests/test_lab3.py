"""Unit tests for Lab 3: Lossless-to-Lossy (L2L) Image Compression System."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to python search path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.lab3.algorithm import (
    dct_2d_fp,
    idct_2d_fp,
    loeffler_dct_2d,
    loeffler_idct_2d,
    fwd_l2l_block_ladder,
    inv_l2l_block_ladder,
    reconstruct_image,
    loeffler_dct_1d,
    loeffler_idct_1d,
)


def test_dct_idct_invertibility():
    """Verify that standard fixed-point 2D DCT and IDCT are close to inverses of each other."""
    np.random.seed(42)
    # Generate a random 8x8 block with grayscale values (scaled to Q10)
    block = np.random.randint(0, 256, (8, 8), dtype=np.int64) << 10
    
    coeffs = dct_2d_fp(block)
    recon = idct_2d_fp(coeffs)
    
    # Check that they match within reasonable fixed-point rounding noise
    diff = np.abs(block - recon) / 1024.0
    assert np.max(diff) < 2.0  # Rounding noise error should be less than 2 LSBs


def test_loeffler_1d_invertibility():
    """Verify that fast Loeffler 1D DCT and 1D IDCT are close to inverses of each other."""
    np.random.seed(42)
    x = np.random.randint(0, 256, 8, dtype=np.int64) << 10
    
    y = loeffler_dct_1d(x)
    x_recon = loeffler_idct_1d(y)
    
    diff = np.abs(x - x_recon) / 1024.0
    assert np.max(diff) < 3.0  # Rounding noise should be small


def test_loeffler_2d_invertibility():
    """Verify that fast Loeffler 2D DCT and 2D IDCT are close to inverses of each other."""
    np.random.seed(42)
    block = np.random.randint(0, 256, (8, 8), dtype=np.int64) << 10
    
    coeffs = loeffler_dct_2d(block)
    recon = loeffler_idct_2d(coeffs)
    
    diff = np.abs(block - recon) / 1024.0
    assert np.max(diff) < 5.0  # Slightly higher due to 1D stage cascades


@pytest.mark.parametrize("use_loeffler", [False, True])
def test_l2l_perfect_reconstruction(use_loeffler):
    """Verify that in Lossless mode (with SIB), the reconstructed image is identical to the original."""
    # Create a small 64x64 synthetic image
    h, w = 64, 64
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            img[r, c] = int(128 + 100 * np.sin(r / 5.0) * np.cos(c / 5.0))
            
    # Forward pass
    blocks_y, states = fwd_l2l_block_ladder(img, use_loeffler=use_loeffler)
    final_state = states[-1]
    
    # Inverse pass with SIB (lossless)
    blocks_x = inv_l2l_block_ladder(blocks_y, final_state, use_sib=True, use_loeffler=use_loeffler)
    recon = reconstruct_image(blocks_x, (h, w))
    
    # Verify bit-true reconstruction
    np.testing.assert_array_equal(img, recon)


@pytest.mark.parametrize("use_loeffler", [False, True])
def test_l2l_lossy_reconstruction(use_loeffler):
    """Verify that in Lossy mode (without SIB), there is reconstruction error (MSE > 0)."""
    h, w = 64, 64
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            img[r, c] = int(128 + 100 * np.sin(r / 5.0) * np.cos(c / 5.0))
            
    # Forward pass
    blocks_y, states = fwd_l2l_block_ladder(img, use_loeffler=use_loeffler)
    final_state = states[-1]
    
    # Inverse pass without SIB (lossy)
    blocks_x = inv_l2l_block_ladder(blocks_y, final_state, use_sib=False, use_loeffler=use_loeffler)
    recon = reconstruct_image(blocks_x, (h, w))
    
    # Verify there is error (MSE > 0)
    mse = np.mean((img.astype(np.float64) - recon.astype(np.float64)) ** 2)
    assert mse > 0.0
