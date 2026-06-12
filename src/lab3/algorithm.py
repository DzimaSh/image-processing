"""L2L Image Compression System algorithm.

Implements the block-ladder (lifting) structural parametrization for 2D DCT-IDCT
transforms in Q10 fixed-point arithmetic with convergent rounding.
Supports both standard matrix-multiplication DCT and fast Loeffler 1D/2D kernels.
"""

import numpy as np

# Fixed-Point constants
SCALE_BITS = 10
SCALE_FACTOR = 1 << SCALE_BITS  # 1024


def scale_and_round(x: np.ndarray, shift: int = 10) -> np.ndarray:
    """Convergent rounding (round-to-nearest-even) to map floats/integers back to fixed-point."""
    return np.round(x / (1 << shift)).astype(np.int64)


# 1D Orthonormal DCT-II Matrix
def get_dct_matrix(M: int = 8) -> np.ndarray:
    """Generate the orthonormal 1D DCT-II matrix of size M x M."""
    C = np.zeros((M, M))
    for k in range(M):
        for n in range(M):
            factor = np.sqrt(1.0 / M) if k == 0 else np.sqrt(2.0 / M)
            C[k, n] = factor * np.cos(np.pi * k * (2 * n + 1) / (2.0 * M))
    return C


# Pre-calculated Q10 fixed-point matrices
C_MAT = get_dct_matrix(8)
D_MAT = C_MAT.T

C_Q10 = np.round(C_MAT * SCALE_FACTOR).astype(np.int64)
D_Q10 = np.round(D_MAT * SCALE_FACTOR).astype(np.int64)


# =====================================================================
# 1. 2D DCT-IDCT via Matrix Multiplication (Orthonormal & High Precision)
# =====================================================================

def dct_2d_fp(block_q: np.ndarray) -> np.ndarray:
    """Perform 2D DCT-II in Q10 fixed-point on an 8x8 block with convergent rounding.

    Formula: C * block * C^T
    """
    # First matrix multiplication: temp = C_Q10 @ block_q
    # Resulting scale is 2^20, so we round back to 2^10
    temp = C_Q10 @ block_q
    temp_q = scale_and_round(temp, SCALE_BITS)
    
    # Second matrix multiplication: out = temp_q @ C_Q10.T
    # C_Q10.T is D_Q10
    out = temp_q @ D_Q10
    return scale_and_round(out, SCALE_BITS)


def idct_2d_fp(block_q: np.ndarray) -> np.ndarray:
    """Perform 2D IDCT-III in Q10 fixed-point on an 8x8 block with convergent rounding.

    Formula: D * block * D^T (D = C^T)
    """
    # First matrix multiplication: temp = D_Q10 @ block_q
    temp = D_Q10 @ block_q
    temp_q = scale_and_round(temp, SCALE_BITS)
    
    # Second matrix multiplication: out = temp_q @ D_Q10.T
    # D_Q10.T is C_Q10
    out = temp_q @ C_Q10
    return scale_and_round(out, SCALE_BITS)


# =====================================================================
# 2. Fast 1D & 2D Loeffler DCT-IDCT Kernels (Low Computational Complexity)
# =====================================================================

# Loeffler rotation constants in Q10 fixed-point
# C_k = round(cos(k * pi / 16) * 1024)
# S_k = round(sin(k * pi / 16) * 1024)
COS_3_16 = 851   # round(cos(3*pi/16) * 1024)
SIN_3_16 = 569   # round(sin(3*pi/16) * 1024)
COS_1_16 = 1004  # round(cos(1*pi/16) * 1024)
SIN_1_16 = 200   # round(sin(1*pi/16) * 1024)
COS_6_16 = 392   # round(cos(6*pi/16) * 1024) (1/sqrt(2))
SIN_6_16 = 946   # round(sin(6*pi/16) * 1024)
COS_4_16 = 724   # round(cos(4*pi/16) * 1024) (1/sqrt(2))
SIN_4_16 = 724

# Output scaling factor 1/(2*sqrt(2)) ~ 0.35355339
# 0.35355339 * 1024 = 362
SCALE_OUT = 362


def loeffler_rot(x: np.ndarray, y: np.ndarray, cos_val: int, sin_val: int) -> tuple[np.ndarray, np.ndarray]:
    """Helper to perform fixed-point vector rotation: x*cos + y*sin, -x*sin + y*cos."""
    rx = scale_and_round(x * cos_val + y * sin_val, SCALE_BITS)
    ry = scale_and_round(-x * sin_val + y * cos_val, SCALE_BITS)
    return rx, ry


def loeffler_dct_1d(x: np.ndarray) -> np.ndarray:
    """Loeffler 8-point 1D DCT-II in fixed point.

    Args:
        x: Array of 8 integers (Q10 scale).

    Returns:
        Array of 8 DCT coefficients (Q10 scale).
    """
    # Stage 1
    a0 = x[0] + x[7]
    a7 = x[0] - x[7]
    a1 = x[1] + x[6]
    a6 = x[1] - x[6]
    a2 = x[2] + x[5]
    a5 = x[2] - x[5]
    a3 = x[3] + x[4]
    a4 = x[3] - x[4]

    # Stage 2
    b0 = a0 + a3
    b3 = a0 - a3
    b1 = a1 + a2
    b2 = a1 - a2
    b4, b7 = loeffler_rot(a4, a7, COS_3_16, SIN_3_16)
    b5, b6 = loeffler_rot(a5, a6, COS_1_16, SIN_1_16)

    # Stage 3
    c0 = b0 + b1
    c1 = b0 - b1
    c2, c3 = loeffler_rot(b2, b3, COS_6_16, SIN_6_16)
    c4 = b4 + b6
    c6 = b4 - b6
    c7 = b7 + b5
    c5 = b7 - b5

    # Stage 4
    d0 = c0
    d1 = c1
    d2 = c2
    d3 = c3
    d4, d7 = loeffler_rot(c4, c7, COS_4_16, SIN_4_16)
    d5, d6 = loeffler_rot(c5, c6, COS_4_16, SIN_4_16)

    # Scale outputs and permute to bit-reversed order
    y = np.empty(8, dtype=np.int64)
    y[0] = scale_and_round(d0 * SCALE_OUT, SCALE_BITS)
    y[4] = scale_and_round(d1 * SCALE_OUT, SCALE_BITS)
    y[2] = scale_and_round(d3 * SCALE_OUT, SCALE_BITS)
    y[6] = scale_and_round(d2 * SCALE_OUT, SCALE_BITS)
    y[1] = scale_and_round(d4 * SCALE_OUT, SCALE_BITS)
    y[7] = scale_and_round(d7 * SCALE_OUT, SCALE_BITS)
    y[5] = scale_and_round(d5 * SCALE_OUT, SCALE_BITS)
    y[3] = scale_and_round(d6 * SCALE_OUT, SCALE_BITS)

    return y


def loeffler_idct_1d(y: np.ndarray) -> np.ndarray:
    """Loeffler 8-point 1D IDCT-III in fixed point.

    Flow is the exact reverse of the forward path.
    """
    # Undo permutation and output scaling (multiply by 2*sqrt(2) = 2896 in Q10)
    SCALE_OUT_INV = 2896
    
    d0 = scale_and_round(y[0] * SCALE_OUT_INV, SCALE_BITS)
    d1 = scale_and_round(y[4] * SCALE_OUT_INV, SCALE_BITS)
    d2 = scale_and_round(y[6] * SCALE_OUT_INV, SCALE_BITS)
    d3 = scale_and_round(y[2] * SCALE_OUT_INV, SCALE_BITS)
    d4 = scale_and_round(y[1] * SCALE_OUT_INV, SCALE_BITS)
    d7 = scale_and_round(y[7] * SCALE_OUT_INV, SCALE_BITS)
    d5 = scale_and_round(y[5] * SCALE_OUT_INV, SCALE_BITS)
    d6 = scale_and_round(y[3] * SCALE_OUT_INV, SCALE_BITS)

    # Stage 4 Inverse
    c0 = d0
    c1 = d1
    c2 = d2
    c3 = d3
    # Rotation inverse is rotation by -theta, i.e. cos(theta), -sin(theta)
    c4, c7 = loeffler_rot(d4, d7, COS_4_16, -SIN_4_16)
    c5, c6 = loeffler_rot(d5, d6, COS_4_16, -SIN_4_16)

    # Stage 3 Inverse
    b0 = scale_and_round((c0 + c1) * 512, SCALE_BITS) # Division by 2
    b1 = scale_and_round((c0 - c1) * 512, SCALE_BITS)
    b2, b3 = loeffler_rot(c2, c3, COS_6_16, -SIN_6_16)
    
    b4 = scale_and_round((c4 + c6) * 512, SCALE_BITS)
    b6 = scale_and_round((c4 - c6) * 512, SCALE_BITS)
    b7 = scale_and_round((c7 + c5) * 512, SCALE_BITS)
    b5 = scale_and_round((c7 - c5) * 512, SCALE_BITS)

    # Stage 2 Inverse
    a0 = scale_and_round((b0 + b3) * 512, SCALE_BITS)
    a3 = scale_and_round((b0 - b3) * 512, SCALE_BITS)
    a1 = scale_and_round((b1 + b2) * 512, SCALE_BITS)
    a2 = scale_and_round((b1 - b2) * 512, SCALE_BITS)
    
    a4, a7 = loeffler_rot(b4, b7, COS_3_16, -SIN_3_16)
    a5, a6 = loeffler_rot(b5, b6, COS_1_16, -SIN_1_16)

    # Stage 1 Inverse
    x = np.empty(8, dtype=np.int64)
    x[0] = scale_and_round((a0 + a7) * 512, SCALE_BITS)
    x[7] = scale_and_round((a0 - a7) * 512, SCALE_BITS)
    x[1] = scale_and_round((a1 + a6) * 512, SCALE_BITS)
    x[6] = scale_and_round((a1 - a6) * 512, SCALE_BITS)
    x[2] = scale_and_round((a2 + a5) * 512, SCALE_BITS)
    x[5] = scale_and_round((a2 - a5) * 512, SCALE_BITS)
    x[3] = scale_and_round((a3 + a4) * 512, SCALE_BITS)
    x[4] = scale_and_round((a3 - a4) * 512, SCALE_BITS)

    return x


def loeffler_dct_2d(block_q: np.ndarray) -> np.ndarray:
    """Perform 2D DCT using separable 1D Loeffler transforms."""
    # Apply to columns
    temp = np.zeros_like(block_q)
    for c in range(8):
        temp[:, c] = loeffler_dct_1d(block_q[:, c])
    # Apply to rows
    out = np.zeros_like(block_q)
    for r in range(8):
        out[r, :] = loeffler_dct_1d(temp[r, :])
    return out


def loeffler_idct_2d(block_q: np.ndarray) -> np.ndarray:
    """Perform 2D IDCT using separable 1D Loeffler transforms."""
    # Apply to rows inverse
    temp = np.zeros_like(block_q)
    for r in range(8):
        temp[r, :] = loeffler_idct_1d(block_q[r, :])
    # Apply to columns inverse
    out = np.zeros_like(block_q)
    for c in range(8):
        out[:, c] = loeffler_idct_1d(temp[:, c])
    return out


# =====================================================================
# 3. L2L Block-Ladder System (Lifting Scheme)
# =====================================================================

def fwd_l2l_block_ladder(img: np.ndarray, use_loeffler: bool = False, M: int = 8) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Forward block-ladder lifting transform.

    Args:
        img: Grayscale source image (values 0-255).
        use_loeffler: Use fast Loeffler DCT-IDCT kernels if True.
        M: Block dimension.

    Returns:
        blocks_y: List of 8x8 transform coefficient blocks (Q10 scale).
        states: List of intermediate states s_0, ..., s_K (Q10 scale).
    """
    h, w = img.shape
    assert h % M == 0 and w % M == 0, f"Image dimensions must be divisible by block size {M}."

    # Choose transform kernel
    dct_fn = loeffler_dct_2d if use_loeffler else dct_2d_fp
    idct_fn = loeffler_idct_2d if use_loeffler else idct_2d_fp

    # Scale image to Q10 fixed-point integers
    img_q = img.astype(np.int64) << SCALE_BITS

    # Decompose image into list of blocks (raster order)
    blocks_x = []
    for r in range(0, h, M):
        for c in range(0, w, M):
            blocks_x.append(img_q[r:r+M, c:c+M].copy())

    num_blocks = len(blocks_x)
    blocks_y = [None] * num_blocks

    # Initialize state chain: s_{-1} = zeros(M, M)
    s_curr = np.zeros((M, M), dtype=np.int64)
    states = [s_curr.copy()]

    # Forward block-ladder steps for each block:
    # 1. u1 = x_n - round(IDCT(s_{n-1}))
    # 2. u2 = s_{n-1} + round(DCT(u1) - u1)
    # 3. y_n = u1 + u2
    # 4. s_n = u2 + round(IDCT(y_n) - y_n)
    for n in range(num_blocks):
        x_n = blocks_x[n]
        s_prev = states[-1]

        # Step 1
        idct_s = idct_fn(s_prev)
        u1 = x_n - idct_s

        # Step 2
        dct_u1 = dct_fn(u1)
        u2 = s_prev + (dct_u1 - u1)

        # Step 3
        y_n = u1 + u2
        blocks_y[n] = y_n

        # Step 4
        idct_yn = idct_fn(y_n)
        s_n = u2 + (idct_yn - y_n)
        states.append(s_n.copy())

    return blocks_y, states


def inv_l2l_block_ladder(blocks_y: list[np.ndarray], final_state: np.ndarray, use_sib: bool = True, use_loeffler: bool = False, M: int = 8) -> list[np.ndarray]:
    """Inverse block-ladder lifting transform.

    Args:
        blocks_y: List of 8x8 transform coefficient blocks (Q10 scale).
        final_state: SIB block (last state s_K from forward pass).
        use_sib: If True, uses the final_state block to achieve perfect reconstruction.
                 If False (lossy mode), initializes the final state to zero.
        use_loeffler: Use fast Loeffler DCT-IDCT kernels if True.
        M: Block dimension.

    Returns:
        blocks_x: Reconstructed spatial blocks (Q10 scale).
    """
    dct_fn = loeffler_dct_2d if use_loeffler else dct_2d_fp
    idct_fn = loeffler_idct_2d if use_loeffler else idct_2d_fp

    num_blocks = len(blocks_y)
    blocks_x = [None] * num_blocks

    # Initialize the backward state chain
    # If use_sib is True, we use the captured SIB (final_state)
    # If use_sib is False, we set it to zero, representing lossy mode (without SIB correction)
    s_curr = final_state.copy() if use_sib else np.zeros((M, M), dtype=np.int64)

    # Inverse block-ladder steps (looping backwards):
    # For n = K-1 down to 0:
    # 1. u2 = s_n - round(IDCT(y_n) - y_n)
    # 2. u1 = y_n - u2
    # 3. s_{n-1} = u2 - round(DCT(u1) - u1)
    # 4. x_n = u1 + round(IDCT(s_{n-1}))
    for n in range(num_blocks - 1, -1, -1):
        y_n = blocks_y[n]

        # Step 1
        idct_yn = idct_fn(y_n)
        u2 = s_curr - (idct_yn - y_n)

        # Step 2
        u1 = y_n - u2

        # Step 3
        dct_u1 = dct_fn(u1)
        s_prev = u2 - (dct_u1 - u1)

        # Step 4
        idct_s_prev = idct_fn(s_prev)
        x_n = u1 + idct_s_prev

        blocks_x[n] = x_n
        s_curr = s_prev.copy()

    return blocks_x


def reconstruct_image(blocks_x: list[np.ndarray], shape: tuple[int, int], M: int = 8) -> np.ndarray:
    """Assemble 8x8 blocks back into a 2D spatial image and scale back to 0-255 range."""
    h, w = shape
    img_recon = np.zeros((h, w), dtype=np.float64)
    idx = 0
    for r in range(0, h, M):
        for c in range(0, w, M):
            img_recon[r:r+M, c:c+M] = blocks_x[idx].astype(np.float64) / SCALE_FACTOR
            idx += 1
    return np.clip(img_recon, 0, 255).astype(np.uint8)
