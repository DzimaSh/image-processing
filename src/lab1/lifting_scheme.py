"""Fixed-point Cohen-Daubechies-Feauveau (CDF) 9/7 Wavelet Lifting Scheme.

This module implements 1D and 2D Discrete Wavelet Transforms (DWT) and their
inverse transforms (IDWT) using Q16 fixed-point arithmetic operations.
Symmetric boundary extensions are used to prevent boundary artifacts.
"""

from typing import Tuple
import numpy as np

# Q16 Fixed-Point Scale Constants (Scaling factor S = 2^16 = 65536)
SCALE_BITS = 16
SCALE_FACTOR = 1 << SCALE_BITS  # 65536
HALF_SCALE = 1 << (SCALE_BITS - 1)  # 32768 (used for rounded division)

# Floating-point CDF 9/7 constants scaled to Q16 integers
# Alpha = -1.586134342059924 -> -103948
ALPHA_Q16 = int(np.round(-1.586134342059924 * SCALE_FACTOR))
# Beta = -0.052980118572961 -> -3472
BETA_Q16 = int(np.round(-0.052980118572961 * SCALE_FACTOR))
# Gamma = 0.882911075530934 -> 57863
GAMMA_Q16 = int(np.round(0.882911075530934 * SCALE_FACTOR))
# Delta = 0.443506852043971 -> 29066
DELTA_Q16 = int(np.round(0.443506852043971 * SCALE_FACTOR))
# K = 1.230174104914001 -> 80620
K_Q16 = int(np.round(1.230174104914001 * SCALE_FACTOR))
# 1/K = 0.812893066115961 -> 53273
K_INV_Q16 = int(np.round(0.812893066115961 * SCALE_FACTOR))


def mul_q16(a: np.ndarray, b: int) -> np.ndarray:
    """Perform vectorized fixed-point multiplication in Q16 format.

    Multiplies a NumPy array by a Q16 integer scalar, applying a round-to-nearest
    operation before right-shifting by 16 bits to preserve scaling.

    Args:
        a: Array of Q16 integers (np.int64).
        b: A scalar Q16 integer coefficient.

    Returns:
        The resulting Q16 array after rounded multiplication.
    """
    # Round-to-nearest is achieved by adding HALF_SCALE before right-shifting
    return np.right_shift(a * b + HALF_SCALE, SCALE_BITS)


def fwd_dwt_1d(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform forward 1D DWT using the fixed-point CDF 9/7 lifting scheme.

    Assumes the signal length is even. Works on arrays of Q16 scaled integers.

    Args:
        signal: 1D NumPy array of signal values in Q16 format.

    Returns:
        A tuple of (smooth_coefficients, detail_coefficients) in Q16 format.
    """
    n = len(signal)
    if n % 2 != 0:
        raise ValueError("Signal length must be even.")

    m = n // 2
    # Split into even and odd components (in-place copying)
    s = signal[0::2].copy().astype(np.int64)
    d = signal[1::2].copy().astype(np.int64)

    # 1. Predict 1 (Alpha)
    s_plus1 = np.empty_like(s)
    s_plus1[:-1] = s[1:]
    s_plus1[-1] = s[-1]  # Symmetric boundary extension x[N] = x[N-2]
    d += mul_q16(s + s_plus1, ALPHA_Q16)

    # 2. Update 1 (Beta)
    d_minus1 = np.empty_like(d)
    d_minus1[1:] = d[:-1]
    d_minus1[0] = d[0]  # Symmetric boundary extension x[-1] = x[1]
    s += mul_q16(d_minus1 + d, BETA_Q16)

    # 3. Predict 2 (Gamma)
    s_plus1[:-1] = s[1:]
    s_plus1[-1] = s[-1]
    d += mul_q16(s + s_plus1, GAMMA_Q16)

    # 4. Update 2 (Delta)
    d_minus1[1:] = d[:-1]
    d_minus1[0] = d[0]
    s += mul_q16(d_minus1 + d, DELTA_Q16)

    # 5. Scaling (K and 1/K)
    s = mul_q16(s, K_Q16)
    d = mul_q16(d, K_INV_Q16)

    return s, d


def inv_dwt_1d(s: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Perform inverse 1D DWT using the fixed-point CDF 9/7 lifting scheme.

    Restores the original signal from Q16 smooth and detail coefficients.

    Args:
        s: Q16 smooth (low-frequency) coefficients.
        d: Q16 detail (high-frequency) coefficients.

    Returns:
        The reconstructed 1D Q16 signal.
    """
    s = s.copy().astype(np.int64)
    d = d.copy().astype(np.int64)

    # 1. Inverse Scaling
    s = mul_q16(s, K_INV_Q16)
    d = mul_q16(d, K_Q16)

    # 2. Inverse Update 2 (Delta)
    d_minus1 = np.empty_like(d)
    d_minus1[1:] = d[:-1]
    d_minus1[0] = d[0]
    s -= mul_q16(d_minus1 + d, DELTA_Q16)

    # 3. Inverse Predict 2 (Gamma)
    s_plus1 = np.empty_like(s)
    s_plus1[:-1] = s[1:]
    s_plus1[-1] = s[-1]
    d -= mul_q16(s + s_plus1, GAMMA_Q16)

    # 4. Inverse Update 1 (Beta)
    d_minus1[1:] = d[:-1]
    d_minus1[0] = d[0]
    s -= mul_q16(d_minus1 + d, BETA_Q16)

    # 5. Inverse Predict 1 (Alpha)
    s_plus1[:-1] = s[1:]
    s_plus1[-1] = s[-1]
    d -= mul_q16(s + s_plus1, ALPHA_Q16)

    # Recombine even and odd components
    reconstructed = np.empty(2 * len(s), dtype=np.int64)
    reconstructed[0::2] = s
    reconstructed[1::2] = d
    return reconstructed


def fwd_dwt_2d_level(matrix: np.ndarray) -> np.ndarray:
    """Perform one level of 2D DWT on a Q16 matrix.

    Transforms rows first, then columns.

    Args:
        matrix: 2D NumPy array of Q16 integers.

    Returns:
        The 2D transformed matrix containing [LL, LH; HL, HH] subbands.
    """
    h, w = matrix.shape
    assert h % 2 == 0 and w % 2 == 0, "Dimensions must be even."

    row_transformed = np.empty_like(matrix)
    # Apply DWT along rows
    for r in range(h):
        s_row, d_row = fwd_dwt_1d(matrix[r])
        row_transformed[r, 0 : w // 2] = s_row
        row_transformed[r, w // 2 : w] = d_row

    col_transformed = np.empty_like(matrix)
    # Apply DWT along columns of row-transformed matrix
    for c in range(w):
        s_col, d_col = fwd_dwt_1d(row_transformed[:, c])
        col_transformed[0 : h // 2, c] = s_col
        col_transformed[h // 2 : h, c] = d_col

    return col_transformed


def inv_dwt_2d_level(matrix: np.ndarray) -> np.ndarray:
    """Perform one level of 2D IDWT on a Q16 matrix.

    Reconstructs columns first, then rows.

    Args:
        matrix: 2D NumPy array of Q16 coefficients.

    Returns:
        The 2D reconstructed matrix in Q16 format.
    """
    h, w = matrix.shape
    assert h % 2 == 0 and w % 2 == 0, "Dimensions must be even."

    col_reconstructed = np.empty_like(matrix)
    # Reconstruct columns
    for c in range(w):
        s_col = matrix[0 : h // 2, c]
        d_col = matrix[h // 2 : h, c]
        col_reconstructed[:, c] = inv_dwt_1d(s_col, d_col)

    row_reconstructed = np.empty_like(matrix)
    # Reconstruct rows
    for r in range(h):
        s_row = col_reconstructed[r, 0 : w // 2]
        d_row = col_reconstructed[r, w // 2 : w]
        row_reconstructed[r] = inv_dwt_1d(s_row, d_row)

    return row_reconstructed


def fwd_dwt_2d(img: np.ndarray, levels: int = 1) -> np.ndarray:
    """Perform multi-level 2D DWT on an image using fixed-point arithmetic.

    Args:
        img: Grayscale input image (2D array, uint8 or float, values in 0-255).
        levels: Number of decomposition levels.

    Returns:
        A 2D array of DWT coefficients in Q16 format (np.int64).
    """
    h, w = img.shape
    if h % (2 ** levels) != 0 or w % (2 ** levels) != 0:
        raise ValueError(
            f"Image dimensions ({h}x{w}) must be divisible by 2^{levels} = {2 ** levels}"
        )

    # Scale the input grayscale image to Q16 format
    coeffs = img.astype(np.int64) << SCALE_BITS

    curr_h, curr_w = h, w
    for _ in range(levels):
        # Extract the current LL sub-band to transform further
        subband = coeffs[0:curr_h, 0:curr_w]
        transformed = fwd_dwt_2d_level(subband)
        coeffs[0:curr_h, 0:curr_w] = transformed
        # Next level operates on the LL subband (top-left quadrant)
        curr_h //= 2
        curr_w //= 2

    return coeffs


def inv_dwt_2d(coeffs: np.ndarray, levels: int = 1) -> np.ndarray:
    """Perform multi-level 2D IDWT to reconstruct an image.

    Args:
        coeffs: 2D array of Q16 DWT coefficients.
        levels: Number of decomposition levels.

    Returns:
        The reconstructed image as a 2D float64 array (values scaled to 0-255).
    """
    h, w = coeffs.shape
    recon = coeffs.copy().astype(np.int64)

    # Start reconstruction from the smallest LL level (e.g. Level L)
    sizes = []
    curr_h, curr_w = h, w
    for _ in range(levels):
        sizes.append((curr_h, curr_w))
        curr_h //= 2
        curr_w //= 2

    # Reverse size traversal
    for level_h, level_w in reversed(sizes):
        subband = recon[0:level_h, 0:level_w]
        reconstructed = inv_dwt_2d_level(subband)
        recon[0:level_h, 0:level_w] = reconstructed

    # Downscale from Q16 to standard float representation [0, 255]
    return recon.astype(np.float64) / SCALE_FACTOR
