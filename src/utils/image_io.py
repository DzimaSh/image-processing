"""Image Input/Output and Visualization utilities.

This module provides helpers to read, write, and visualize grayscale images
and their wavelet transform coefficients using OpenCV.
"""

import os
import cv2
import numpy as np


def load_image_grayscale(path: str) -> np.ndarray:
    """Load an image from disk and convert it to grayscale.

    Args:
        path: Path to the image file.

    Returns:
        The grayscale image as a 2D NumPy array (uint8).

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the file is not a valid image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at path: {path}")

    # Read image as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at path: {path}. Unsupported format?")

    return img


def save_image(path: str, img: np.ndarray) -> None:
    """Save an image array to disk.

    Ensures parent directories exist before saving. Clips the image to
    0-255 and casts to uint8.

    Args:
        path: Destination file path.
        img: The image as a 2D NumPy array.
    """
    # Ensure directory exists
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Standardize image array for saving (clip to 0-255 and cast to uint8)
    img_clipped = np.clip(img, 0, 255).astype(np.uint8)
    success = cv2.imwrite(path, img_clipped)
    if not success:
        raise IOError(f"Failed to write image to: {path}")


def create_dwt_visualization(coeffs: np.ndarray, levels: int = 1) -> np.ndarray:
    """Create a visually appealing representation of DWT coefficients.

    For visualization, the approximation subband (LL) is scaled directly to 0-255,
    while the detail subbands (LH, HL, HH) are scaled, centered at 128 (to show
    negative details as darker, positive as brighter), and clipped to uint8.

    Args:
        coeffs: The DWT coefficients matrix (2D float64/int64 array).
        levels: Number of levels of DWT decomposition.

    Returns:
        A 2D NumPy array (uint8) ready to be saved as an image.
    """
    h, w = coeffs.shape
    vis = coeffs.copy().astype(np.float64)

    # Recursively process levels to map coefficients to [0, 255]
    curr_h, curr_w = h, w
    for _ in range(levels):
        half_h, half_w = curr_h // 2, curr_w // 2

        # 1. LL Sub-band (Top-Left)
        ll = vis[0:half_h, 0:half_w]
        ll_min, ll_max = ll.min(), ll.max()
        if ll_max - ll_min > 1e-5:
            vis[0:half_h, 0:half_w] = (ll - ll_min) / (ll_max - ll_min) * 255.0
        else:
            vis[0:half_h, 0:half_w] = np.clip(ll, 0, 255)

        # 2. Detail Sub-bands: LH, HL, HH
        # We map these details such that 0 is represented as 128,
        # negative details are darker, and positive details are brighter.
        for slices in [
            (slice(0, half_h), slice(half_w, curr_w)),  # LH (Top-Right)
            (slice(half_h, curr_h), slice(0, half_w)),  # HL (Bottom-Left)
            (slice(half_h, curr_h), slice(half_w, curr_w)),  # HH (Bottom-Right)
        ]:
            detail = vis[slices]
            max_abs = np.max(np.abs(detail))
            if max_abs > 1e-5:
                # Map [-max_abs, max_abs] to [0, 255] centered at 128
                vis[slices] = 128.0 + (detail / max_abs) * 127.0
            else:
                vis[slices] = 128.0

        curr_h, curr_w = half_h, half_w

    # Draw grid boundaries between subbands to enhance premium design presentation
    vis_img = np.clip(vis, 0, 255).astype(np.uint8)
    curr_h, curr_w = h, w
    for _ in range(levels):
        half_h, half_w = curr_h // 2, curr_w // 2
        # Horizontal boundary line
        vis_img[half_h - 1 : half_h + 1, 0:curr_w] = 255
        # Vertical boundary line
        vis_img[0:curr_h, half_w - 1 : half_w + 1] = 255
        curr_h, curr_w = half_h, half_w

    return vis_img
