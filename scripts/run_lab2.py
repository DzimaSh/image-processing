#!/usr/bin/env python3
"""CLI Script to execute Lab 2: 2D Non-Separable Wavelet Transforms.

Loads an image, applies either the Reversible Quaternionic Paraunitary Filter Bank
or the 2D Non-Separable CDF 9/7 DWT, performs reconstruction, calculates metrics,
and saves result visualizations.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in Python search path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.lab2 import (
    fwd_quaternionic_lifting_2d,
    inv_quaternionic_lifting_2d,
    fwd_non_separable_cdf97_2d,
    inv_non_separable_cdf97_2d,
)
from src.shared.metrics import calculate_mse, calculate_psnr
from src.shared.image_io import (
    load_image_grayscale,
    save_image,
    create_dwt_visualization,
)


def generate_synthetic_image(path: str) -> None:
    """Generate a highly structured synthetic image for testing/calibration.

    Creates a 256x256 image with a checkerboard pattern, smooth gradients,
    and a concentric circle pattern to test high-frequency details.
    """
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)

    # 1. Checkerboard pattern
    for r in range(size):
        for c in range(size):
            if ((r // 32) + (c // 32)) % 2 == 0:
                img[r, c] = 200
            else:
                img[r, c] = 50

    # 2. Concentric circular gradient in the center
    y, x = np.ogrid[:size, :size]
    center = size // 2
    r_sq = (x - center) ** 2 + (y - center) ** 2
    circle_mask = r_sq < (80 ** 2)
    img[circle_mask] = (np.cos(np.sqrt(r_sq[circle_mask]) / 10.0) * 127.0 + 128.0).astype(np.uint8)

    # 3. Add diagonal sharp line detail
    for i in range(size):
        img[i, i] = 255
        img[i, size - 1 - i] = 0

    save_image(path, img)
    print(f"[Info] Generated a beautiful synthetic image at: {path}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="2D Non-Separable Wavelet Transforms & Paraunitary Filter Banks (Lab 2)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to the input image. If omitted, a synthetic image is generated.",
    )
    parser.add_argument(
        "--transform",
        type=str,
        choices=["quaternionic", "non-separable-cdf"],
        default="quaternionic",
        help="Type of 2D non-separable transform to execute (default: quaternionic).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(project_root / "data" / "output"),
        help="Directory to save output files.",
    )

    args = parser.parse_args()

    # Determine input image
    if args.image:
        input_path = args.image
    else:
        input_dir = project_root / "data"
        os.makedirs(input_dir, exist_ok=True)
        synthetic_path = input_dir / "sample_synthetic.png"
        if not synthetic_path.exists():
            generate_synthetic_image(str(synthetic_path))
        input_path = str(synthetic_path)

    print(f"\n========================================================")
    print(f"Executing Lab 2: 2D Non-Separable Transform")
    print(f"Algorithm Type: {args.transform.upper()}")
    print(f"Input Image:    {input_path}")
    print(f"========================================================")

    # 1. Load image
    try:
        orig_img = load_image_grayscale(input_path)
    except Exception as e:
        print(f"[Error] Failed to load image: {e}")
        sys.exit(1)

    print(f"Original image shape: {orig_img.shape} (Grayscale)")

    # Assert dimensions are even (2D decomposition requirement)
    h, w = orig_img.shape
    if h % 2 != 0 or w % 2 != 0:
        new_h = (h // 2) * 2
        new_w = (w // 2) * 2
        print(
            f"[Warning] Image size {h}x{w} is not even. "
            f"Cropping image to {new_h}x{new_w}."
        )
        orig_img = orig_img[:new_h, :new_w]

    # 2. Perform Forward Wavelet Transform
    print("\n[Step 1] Performing forward transform...")
    if args.transform == "quaternionic":
        coeffs = fwd_quaternionic_lifting_2d(orig_img)
    else:
        coeffs = fwd_non_separable_cdf97_2d(orig_img)

    # 3. Create Wavelet Coefficient Visualization
    print("[Step 2] Creating transform coefficient visualization...")
    vis_img = create_dwt_visualization(coeffs, levels=1)
    vis_path = os.path.join(args.out_dir, f"lab2_vis_{args.transform}.png")
    try:
        save_image(vis_path, vis_img)
        print(f" -> Saved DWT visualization to: {vis_path}")
    except Exception as e:
        print(f"[Error] Failed to save transform visualization: {e}")

    # 4. Perform Inverse Transform
    print("[Step 3] Performing inverse transform...")
    if args.transform == "quaternionic":
        recon_img = inv_quaternionic_lifting_2d(coeffs)
    else:
        recon_img = inv_non_separable_cdf97_2d(coeffs)

    # 5. Save Reconstructed Image
    recon_path = os.path.join(args.out_dir, f"lab2_recon_{args.transform}.png")
    try:
        save_image(recon_path, recon_img)
        print(f" -> Saved reconstructed image to: {recon_path}")
    except Exception as e:
        print(f"[Error] Failed to save reconstructed image: {e}")

    # 6. Calculate Metrics
    print("\n[Step 4] Analyzing reconstruction accuracy...")
    mse = calculate_mse(orig_img, recon_img)
    psnr = calculate_psnr(orig_img, recon_img)

    print(f"\nReconstruction Analysis Summary ({args.transform.upper()}):")
    print(f"----------------------------------------")
    print(f"Mean Squared Error (MSE):       {mse:.6f}")
    if np.isinf(psnr):
        print(f"Peak Signal-to-Noise Ratio:     Infinite dB (Perfect Reconstruction)")
    else:
        print(f"Peak Signal-to-Noise Ratio:     {psnr:.3f} dB")
    print(f"----------------------------------------")
    print("Execution completed successfully!\n")


if __name__ == "__main__":
    main()
