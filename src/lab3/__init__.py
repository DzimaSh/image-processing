"""Lab 3: Lossless-to-Lossy (L2L) Image Compression System.

This module implements:
1. Block-ladder structural parametrization using 2D DCT-IDCT block transforms.
2. Side Information Block (SIB) for perfect reconstruction and checkerboard removal.
3. Q10 Fixed-Point arithmetic with convergent rounding.
"""

from .algorithm import (
    fwd_l2l_block_ladder,
    inv_l2l_block_ladder,
    reconstruct_image,
    scale_and_round,
    dct_2d_fp,
    idct_2d_fp,
)

__all__ = [
    "fwd_l2l_block_ladder",
    "inv_l2l_block_ladder",
    "reconstruct_image",
    "scale_and_round",
    "dct_2d_fp",
    "idct_2d_fp",
]
