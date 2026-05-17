"""2D Non-Separable Wavelet Transforms and Paraunitary Filter Banks.

This module implements:
1. Reversible 2D Non-Separable Quaternionic Paraunitary Filter Bank (Q-PUFB).
2. 2D Non-Separable CDF 9/7 Lifting Scheme.

Both transforms are implemented using Q16 fixed-point arithmetic for speed,
numerical precision, and perfect reversibility.
"""

from typing import Tuple
import numpy as np

# Q16 Fixed-Point Scale Constants (Scaling factor S = 2^16 = 65536)
SCALE_BITS = 16
SCALE_FACTOR = 1 << SCALE_BITS  # 65536
HALF_SCALE = 1 << (SCALE_BITS - 1)  # 32768 (used for rounded division)


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
    return np.right_shift(a * b + HALF_SCALE, SCALE_BITS)


# =====================================================================
# 1. 2D Non-Separable Quaternionic Paraunitary Filter Bank (Q-PUFB)
# =====================================================================

# Default unit-norm quaternion: Q = [0.8, 0.2, 0.4, 0.4]
# Gives exact rational coefficients for block-lifting matrices:
# fa = -0.5, fb = 0.0
# ha = 0.0, hb = -0.5
# ga = 0.4, gb = 0.4

FA_Q16 = int(np.round(-0.5 * SCALE_FACTOR))  # -32768
FB_Q16 = 0
HA_Q16 = 0
HB_Q16 = int(np.round(-0.5 * SCALE_FACTOR))  # -32768
GA_Q16 = int(np.round(0.4 * SCALE_FACTOR))   # 26214
GB_Q16 = int(np.round(0.4 * SCALE_FACTOR))   # 26214


def fwd_quaternionic_lifting_2d(img: np.ndarray) -> np.ndarray:
    """Perform forward 2D Quaternionic Paraunitary Filter Bank transform.

    Splits the image into 4 subgrids (ee, eo, oe, oo) and applies the
    quaternionic multiplier block-lifting stages factorized as U * L * V.

    Args:
        img: Grayscale input image (2D uint8/float64, values 0-255).

    Returns:
        A 2D array of transformed coefficients in Q16 format (np.int64).
    """
    h, w = img.shape
    assert h % 2 == 0 and w % 2 == 0, "Dimensions must be even for 2D decomposition."

    # Scale the input to Q16 format
    coeffs = (img.astype(np.int64) << SCALE_BITS)

    # Decompose into 4 subgrids
    ee = coeffs[0::2, 0::2].copy()
    eo = coeffs[0::2, 1::2].copy()
    oe = coeffs[1::2, 0::2].copy()
    oo = coeffs[1::2, 1::2].copy()

    # Form 2-component vector signals:
    # vA = [ee, eo]^T
    # vB = [oe, oo]^T

    # 1. Block-Lifting Step V(Q): vA = vA + H(Q) * vB
    # H(Q) = [[0, -0.5], [-0.5, 0]]
    # So: ee += -0.5 * oo
    #     eo += -0.5 * oe
    ee += mul_q16(oo, HB_Q16)
    eo += mul_q16(oe, HB_Q16)

    # 2. Block-Lifting Step L(Q): vB = vB + G(Q) * vA
    # G(Q) = S(Q) = [[0.4, 0.4], [0.4, -0.4]]
    # So: oe += 0.4 * ee + 0.4 * eo
    #     oo += 0.4 * ee - 0.4 * eo
    oe += mul_q16(ee, GA_Q16) + mul_q16(eo, GB_Q16)
    oo += mul_q16(ee, GA_Q16) - mul_q16(eo, GB_Q16)

    # 3. Block-Lifting Step U(Q): vA = vA + F(Q) * vB
    # F(Q) = [[-0.5, 0], [0, 0.5]]
    # So: ee += -0.5 * oe
    #     eo += 0.5 * oo
    ee += mul_q16(oe, FA_Q16)
    eo -= mul_q16(oo, FA_Q16)  # Subtract since -(-0.5) is +0.5

    # Assemble back into transformed representation [LL, LH; HL, HH]
    out = np.empty_like(coeffs)
    hh, hw = h // 2, w // 2
    out[0:hh, 0:hw] = ee
    out[0:hh, hw:w] = eo
    out[hh:h, 0:hw] = oe
    out[hh:h, hw:w] = oo
    return out


def inv_quaternionic_lifting_2d(coeffs: np.ndarray) -> np.ndarray:
    """Perform inverse 2D Quaternionic Paraunitary Filter Bank transform.

    Reconstructs the original grayscale image from quaternionic DWT coefficients
    by applying the inverse block-lifting steps in exact reverse order.

    Args:
        coeffs: 2D array of Q16 transformed coefficients.

    Returns:
        The reconstructed image as a 2D float64 array (values scaled to 0-255).
    """
    h, w = coeffs.shape
    hh, hw = h // 2, w // 2

    # Extract subgrids
    ee = coeffs[0:hh, 0:hw].copy()
    eo = coeffs[0:hh, hw:w].copy()
    oe = coeffs[hh:h, 0:hw].copy()
    oo = coeffs[hh:h, hw:w].copy()

    # 1. Inverse Step U(Q): vA = vA - F(Q) * vB
    # ee -= -0.5 * oe
    # eo -= 0.5 * oo
    ee -= mul_q16(oe, FA_Q16)
    eo += mul_q16(oo, FA_Q16)

    # 2. Inverse Step L(Q): vB = vB - G(Q) * vA
    # oe -= 0.4 * ee + 0.4 * eo
    # oo -= 0.4 * ee - 0.4 * eo
    oe -= mul_q16(ee, GA_Q16) + mul_q16(eo, GB_Q16)
    oo -= mul_q16(ee, GA_Q16) - mul_q16(eo, GB_Q16)

    # 3. Inverse Step V(Q): vA = vA - H(Q) * vB
    # ee -= -0.5 * oo
    # eo -= -0.5 * oe
    ee -= mul_q16(oo, HB_Q16)
    eo -= mul_q16(oe, HB_Q16)

    # Recombine subgrids into final image
    recon = np.empty_like(coeffs)
    recon[0::2, 0::2] = ee
    recon[0::2, 1::2] = eo
    recon[1::2, 0::2] = oe
    recon[1::2, 1::2] = oo

    # Convert back to standard representation [0-255]
    return recon.astype(np.float64) / SCALE_FACTOR


# =====================================================================
# 2. 2D Non-Separable CDF 9/7 Wavelet Transform
# =====================================================================

# Floating point constants for CDF 9/7
ALPHA_Q16 = int(np.round(-1.586134342059924 * SCALE_FACTOR))
BETA_Q16 = int(np.round(-0.052980118572961 * SCALE_FACTOR))
GAMMA_Q16 = int(np.round(0.882911075530934 * SCALE_FACTOR))
DELTA_Q16 = int(np.round(0.443506852043971 * SCALE_FACTOR))
K_Q16 = int(np.round(1.230174104914001 * SCALE_FACTOR))
K_INV_Q16 = int(np.round(0.812893066115961 * SCALE_FACTOR))


def pad_symmetric_2d(arr: np.ndarray) -> np.ndarray:
    """Pad a 2D array symmetrically by 1 pixel on all sides."""
    return np.pad(arr, pad_width=1, mode="symmetric")


def fwd_non_separable_cdf97_2d(img: np.ndarray) -> np.ndarray:
    """Perform forward 2D Non-Separable CDF 9/7 DWT using direct 2D lifting steps.

    Args:
        img: Grayscale input image (2D uint8/float64, values 0-255).

    Returns:
        A 2D array of DWT coefficients in Q16 format (np.int64).
    """
    h, w = img.shape
    assert h % 2 == 0 and w % 2 == 0, "Dimensions must be even."

    # Scale the input grayscale image to Q16 format
    coeffs = img.astype(np.int64) << SCALE_BITS

    # Split into 4 subgrids
    ee = coeffs[0::2, 0::2].copy()
    eo = coeffs[0::2, 1::2].copy()
    oe = coeffs[1::2, 0::2].copy()
    oo = coeffs[1::2, 1::2].copy()

    # Step 1: Predict 1 (Alpha)
    # ee_padded is needed for right and bottom neighbor elements
    ee_pad = pad_symmetric_2d(ee)
    
    # ee[i, j+1] and ee[i+1, j]
    ee_right = ee_pad[1:-1, 2:]
    ee_down = ee_pad[2:, 1:-1]
    
    eo += mul_q16(ee + ee_right, ALPHA_Q16)
    oe += mul_q16(ee + ee_down, ALPHA_Q16)
    
    # oo is predicted from updated eo and oe
    eo_pad = pad_symmetric_2d(eo)
    oe_pad = pad_symmetric_2d(oe)
    eo_down = eo_pad[2:, 1:-1]
    oe_right = oe_pad[1:-1, 2:]
    
    oo += mul_q16(eo + eo_down + oe + oe_right, ALPHA_Q16)

    # Step 2: Update 1 (Beta)
    eo_pad = pad_symmetric_2d(eo)
    oe_pad = pad_symmetric_2d(oe)
    
    # eo[i, j-1] and oe[i-1, j]
    eo_left = eo_pad[1:-1, :-2]
    oe_up = oe_pad[:-2, 1:-1]
    
    ee += mul_q16(eo_left + eo + oe_up + oe, BETA_Q16)

    # Step 3: Predict 2 (Gamma)
    ee_pad = pad_symmetric_2d(ee)
    ee_right = ee_pad[1:-1, 2:]
    ee_down = ee_pad[2:, 1:-1]
    
    eo += mul_q16(ee + ee_right, GAMMA_Q16)
    oe += mul_q16(ee + ee_down, GAMMA_Q16)
    
    eo_pad = pad_symmetric_2d(eo)
    oe_pad = pad_symmetric_2d(oe)
    eo_down = eo_pad[2:, 1:-1]
    oe_right = oe_pad[1:-1, 2:]
    
    oo += mul_q16(eo + eo_down + oe + oe_right, GAMMA_Q16)

    # Step 4: Update 2 (Delta)
    eo_pad = pad_symmetric_2d(eo)
    oe_pad = pad_symmetric_2d(oe)
    
    eo_left = eo_pad[1:-1, :-2]
    oe_up = oe_pad[:-2, 1:-1]
    
    ee += mul_q16(eo_left + eo + oe_up + oe, DELTA_Q16)

    # Step 5: Scaling
    # Scale LL (ee) by K^2, and HH (oo) by 1/K^2
    k2 = int(np.round(1.230174104914001**2 * SCALE_FACTOR))
    k2_inv = int(np.round(0.812893066115961**2 * SCALE_FACTOR))
    
    ee = mul_q16(ee, k2)
    oo = mul_q16(oo, k2_inv)

    # Assemble into standard subband layout [LL, LH; HL, HH]
    out = np.empty_like(coeffs)
    hh, hw = h // 2, w // 2
    out[0:hh, 0:hw] = ee
    out[0:hh, hw:w] = eo
    out[hh:h, 0:hw] = oe
    out[hh:h, hw:w] = oo
    return out


def inv_non_separable_cdf97_2d(coeffs: np.ndarray) -> np.ndarray:
    """Perform inverse 2D Non-Separable CDF 9/7 DWT to reconstruct the image.

    Args:
        coeffs: 2D array of DWT coefficients in Q16 format.

    Returns:
        The reconstructed image as a 2D float64 array (values scaled to 0-255).
    """
    h, w = coeffs.shape
    hh, hw = h // 2, w // 2

    # Extract subbands
    ee = coeffs[0:hh, 0:hw].copy()
    eo = coeffs[0:hh, hw:w].copy()
    oe = coeffs[hh:h, 0:hw].copy()
    oo = coeffs[hh:h, hw:w].copy()

    # Step 1: Inverse Scaling
    k2 = int(np.round(1.230174104914001**2 * SCALE_FACTOR))
    k2_inv = int(np.round(0.812893066115961**2 * SCALE_FACTOR))
    
    ee = mul_q16(ee, k2_inv)
    oo = mul_q16(oo, k2)

    # Step 2: Inverse Update 2 (Delta)
    eo_pad = pad_symmetric_2d(eo)
    oe_pad = pad_symmetric_2d(oe)
    
    eo_left = eo_pad[1:-1, :-2]
    oe_up = oe_pad[:-2, 1:-1]
    
    ee -= mul_q16(eo_left + eo + oe_up + oe, DELTA_Q16)

    # Step 3: Inverse Predict 2 (Gamma)
    eo_pad = pad_symmetric_2d(eo)
    oe_pad = pad_symmetric_2d(oe)
    eo_down = eo_pad[2:, 1:-1]
    oe_right = oe_pad[1:-1, 2:]
    
    oo -= mul_q16(eo + eo_down + oe + oe_right, GAMMA_Q16)
    
    ee_pad = pad_symmetric_2d(ee)
    ee_right = ee_pad[1:-1, 2:]
    ee_down = ee_pad[2:, 1:-1]
    
    eo -= mul_q16(ee + ee_right, GAMMA_Q16)
    oe -= mul_q16(ee + ee_down, GAMMA_Q16)

    # Step 4: Inverse Update 1 (Beta)
    eo_pad = pad_symmetric_2d(eo)
    oe_pad = pad_symmetric_2d(oe)
    
    eo_left = eo_pad[1:-1, :-2]
    oe_up = oe_pad[:-2, 1:-1]
    
    ee -= mul_q16(eo_left + eo + oe_up + oe, BETA_Q16)

    # Step 5: Inverse Predict 1 (Alpha)
    eo_pad = pad_symmetric_2d(eo)
    oe_pad = pad_symmetric_2d(oe)
    eo_down = eo_pad[2:, 1:-1]
    oe_right = oe_pad[1:-1, 2:]
    
    oo -= mul_q16(eo + eo_down + oe + oe_right, ALPHA_Q16)
    
    ee_pad = pad_symmetric_2d(ee)
    ee_right = ee_pad[1:-1, 2:]
    ee_down = ee_pad[2:, 1:-1]
    
    eo -= mul_q16(ee + ee_right, ALPHA_Q16)
    oe -= mul_q16(ee + ee_down, ALPHA_Q16)

    # Recombine subgrids
    recon = np.empty_like(coeffs)
    recon[0::2, 0::2] = ee
    recon[0::2, 1::2] = eo
    recon[1::2, 0::2] = oe
    recon[1::2, 1::2] = oo

    return recon.astype(np.float64) / SCALE_FACTOR
