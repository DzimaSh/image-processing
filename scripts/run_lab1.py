#!/usr/bin/env python3
"""CLI Script to execute Lab 1: CDF 9/7 Wavelet Transform using Fixed-Point Lifting.

Loads a grayscale image, performs the forward and inverse transforms,
displays the MSE and PSNR, and saves both the reconstructed image and
the DWT coefficient visualization to data/output/.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Ensure the project root is in the Python search path to resolve 'src' imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.lab1.lifting_scheme import fwd_dwt_2d, inv_dwt_2d
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
    # Overwrite center with smooth radial gradient
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
        description="CDF 9/7 Wavelet Transform via Fixed-Point Lifting Scheme (Lab 1)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to the input image. If omitted, a synthetic image is generated.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=1,
        help="Number of multi-level decomposition iterations (default: 1).",
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

    print("\n========================================================")
    print(f"Executing CDF 9/7 Fixed-Point DWT (Levels: {args.levels})")
    print(f"Input Image: {input_path}")
    print("========================================================")

    # 1. Load image
    try:
        orig_img = load_image_grayscale(input_path)
    except Exception as e:
        print(f"[Error] Failed to load image: {e}")
        sys.exit(1)

    print(f"Original image shape: {orig_img.shape} (Grayscale)")

    # Adjust image dimensions to be divisible by 2^levels if needed
    h, w = orig_img.shape
    divisor = 2 ** args.levels
    if h % divisor != 0 or w % divisor != 0:
        new_h = (h // divisor) * divisor
        new_w = (w // divisor) * divisor
        print(
            f"[Warning] Image size {h}x{w} not divisible by 2^{args.levels}. "
            f"Cropping image to {new_h}x{new_w}."
        )
        orig_img = orig_img[:new_h, :new_w]

    # 2. Perform Forward Wavelet Transform (Q16 Fixed-Point)
    print("\n[Step 1] Performing forward 2D DWT...")
    coeffs = fwd_dwt_2d(orig_img, levels=args.levels)

    # 3. Create Wavelet Coefficient Visualization
    print("[Step 2] Creating wavelet sub-band visualization...")
    vis_img = create_dwt_visualization(coeffs, levels=args.levels)
    vis_path = os.path.join(args.out_dir, f"dwt_vis_level_{args.levels}.png")
    try:
        save_image(vis_path, vis_img)
        print(f" -> Saved DWT visualization to: {vis_path}")
    except Exception as e:
        print(f"[Error] Failed to save DWT visualization: {e}")

    # 4. Perform Inverse Wavelet Transform
    print("[Step 3] Performing inverse 2D DWT...")
    recon_img = inv_dwt_2d(coeffs, levels=args.levels)

    # 5. Save Reconstructed Image
    recon_path = os.path.join(args.out_dir, "reconstructed.png")
    try:
        save_image(recon_path, recon_img)
        print(f" -> Saved reconstructed image to: {recon_path}")
    except Exception as e:
        print(f"[Error] Failed to save reconstructed image: {e}")

    # 6. Calculate Metrics
    print("\n[Step 4] Analyzing reconstruction accuracy...")
    mse = calculate_mse(orig_img, recon_img)
    psnr = calculate_psnr(orig_img, recon_img)

    print("\nReconstruction Analysis Summary:")
    print("----------------------------------------")
    print(f"Mean Squared Error (MSE):       {mse:.6f}")
    if np.isinf(psnr):
        print("Peak Signal-to-Noise Ratio:     Infinite dB (Perfect Reconstruction)")
    else:
        print(f"Peak Signal-to-Noise Ratio:     {psnr:.3f} dB")
    print("----------------------------------------")
    print("Execution completed successfully!\n")


if __name__ == "__main__":
    main()
