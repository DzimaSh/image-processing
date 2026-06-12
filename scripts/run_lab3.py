#!/usr/bin/env python
"""CLI application to run and evaluate the Lab 3 L2L Image Compression System.

This script executes the block-ladder DCT-IDCT transform with and without SIB,
evaluates the reconstruction fidelity (PSNR/MSE), and saves the results.
"""

import argparse
import os
import cv2
import sys
from pathlib import Path
import numpy as np

# Ensure the project root is in the Python search path to resolve 'src' imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.lab3.algorithm import (
    fwd_l2l_block_ladder,
    inv_l2l_block_ladder,
    reconstruct_image,
)
from src.shared.metrics import calculate_mse, calculate_psnr
from src.shared.image_io import load_image_grayscale, save_image


def generate_synthetic_image(h=256, w=256):
    """Generate a high-quality synthetic calibration image with gradients and structures."""
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            val = 128 + 127 * np.sin(r / 12.0) * np.cos(c / 12.0)
            # Add some high-frequency chess pattern in the center
            if 80 <= r < 176 and 80 <= c < 176:
                if ((r // 8) + (c // 8)) % 2 == 0:
                    val = np.clip(val + 40, 0, 255)
                else:
                    val = np.clip(val - 40, 0, 255)
            img[r, c] = int(val)
    return img


def main():
    parser = argparse.ArgumentParser(description="L2L block-ladder DCT-IDCT image compression system")
    parser.add_argument("--image", type=str, default=None, help="Path to input grayscale image. If not set, generates synthetic image.")
    parser.add_argument("--use-loeffler", action="store_true", help="Use fast Loeffler 1D/2D DCT-IDCT kernels instead of standard matrices.")
    parser.add_argument("--block-size", type=int, default=8, help="Block size M x M (default 8).")
    args = parser.parse_args()

    # 1. Load or generate image
    if args.image:
        print(f"Loading image from: {args.image}")
        try:
            img = load_image_grayscale(args.image)
        except Exception as e:
            print(f"Error loading image: {e}. Generating synthetic image instead.")
            img = generate_synthetic_image()
    else:
        print("No input image specified. Generating synthetic calibration image.")
        img = generate_synthetic_image()

    h, w = img.shape
    print(f"Image Resolution: {h}x{w} pixels")
    print(f"Block Size: {args.block_size}x{args.block_size}")
    print(f"DCT Kernel: {'Loeffler Fast 1D/2D' if args.use_loeffler else 'Standard Matrix Multiply'}")

    # 2. Run Forward transform
    print("\nRunning Forward L2L block-ladder transform...")
    blocks_y, states = fwd_l2l_block_ladder(img, use_loeffler=args.use_loeffler, M=args.block_size)
    final_state = states[-1]

    # 3. Reconstruct WITH SIB (Lossless mode)
    print("Reconstructing WITH SIB (Lossless mode)...")
    blocks_x_lossless = inv_l2l_block_ladder(
        blocks_y, final_state, use_sib=True, use_loeffler=args.use_loeffler, M=args.block_size
    )
    img_lossless = reconstruct_image(blocks_x_lossless, (h, w), M=args.block_size)

    # 4. Reconstruct WITHOUT SIB (Lossy mode)
    print("Reconstructing WITHOUT SIB (Lossy mode)...")
    blocks_x_lossy = inv_l2l_block_ladder(
        blocks_y, final_state, use_sib=False, use_loeffler=args.use_loeffler, M=args.block_size
    )
    img_lossy = reconstruct_image(blocks_x_lossy, (h, w), M=args.block_size)

    # 5. Calculate metrics
    psnr_lossless = calculate_psnr(img, img_lossless)
    mse_lossless = calculate_mse(img, img_lossless)

    psnr_lossy = calculate_psnr(img, img_lossy)
    mse_lossy = calculate_mse(img, img_lossy)

    # 6. Print Comparison Table
    print("\n" + "=" * 50)
    print(f"{'Reconstruction Mode':<25} | {'PSNR (dB)':<10} | {'MSE':<10}")
    print("-" * 50)
    print(f"{'Lossless (With SIB)':<25} | {str(psnr_lossless):<10} | {mse_lossless:<10.6f}")
    print(f"{'Lossy (Without SIB)':<25} | {psnr_lossy:<10.4f} | {mse_lossy:<10.6f}")
    print("=" * 50)

    # 7. Save outputs
    out_dir = "data/output"
    os.makedirs(out_dir, exist_ok=True)
    
    orig_path = os.path.join(out_dir, "lab3_orig.png")
    lossless_path = os.path.join(out_dir, "lab3_lossless.png")
    lossy_path = os.path.join(out_dir, "lab3_lossy.png")
    
    save_image(orig_path, img)
    save_image(lossless_path, img_lossless)
    save_image(lossy_path, img_lossy)

    print(f"\nSaved original image to: {orig_path}")
    print(f"Saved lossless reconstructed image to: {lossless_path}")
    print(f"Saved lossy reconstructed image (checkerboard) to: {lossy_path}")
    print("Verification complete.")


if __name__ == "__main__":
    main()
