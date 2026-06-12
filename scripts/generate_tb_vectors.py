import numpy as np
import sys
from pathlib import Path

# Add project root to python search path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.lab3.algorithm import dct_2d_fp, idct_2d_fp

def generate_vectors():
    np.random.seed(42)
    num_blocks = 5
    
    # We will generate blocks representing pixel values.
    # To fit in ap_fixed<12, 2>, we scale the pixels down by 256.0.
    # In Q10, this means we multiply the pixel values [0, 255] by 4.
    blocks_x = []
    for b in range(num_blocks):
        # Let's create different pattern blocks
        if b == 0:
            # Gradient block
            block = np.fromfunction(lambda r, c: 10 + r * 15 + c * 10, (8, 8), dtype=np.int64)
        elif b == 1:
            # Flat block
            block = np.full((8, 8), 128, dtype=np.int64)
        elif b == 2:
            # High frequency checkerboard
            block = np.fromfunction(lambda r, c: 128 + 50 * (((r + c) % 2) * 2 - 1), (8, 8), dtype=np.int64)
        else:
            # Random block
            block = np.random.randint(0, 256, (8, 8), dtype=np.int64)
        
        # Scale to Q10 with division by 256, i.e., multiply by 4
        blocks_x.append(block * 4)
        
    # Forward ladder step logic:
    # 1. u1 = x_n - IDCT(s_prev)
    # 2. u2 = s_prev + DCT(u1) - u1
    # 3. y_n = u1 + u2
    # 4. s_n = u2 + IDCT(y_n) - y_n
    
    s_prev = np.zeros((8, 8), dtype=np.int64)
    
    out_path = project_root / "hardware" / "src" / "tb_vectors.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        f.write(f"NUM_BLOCKS {num_blocks}\n")
        
        for b in range(num_blocks):
            x_n = blocks_x[b]
            
            # Step 1
            idct_s = idct_2d_fp(s_prev)
            u1 = x_n - idct_s
            
            # Step 2
            dct_u1 = dct_2d_fp(u1)
            u2 = s_prev + (dct_u1 - u1)
            
            # Step 3
            y_n = u1 + u2
            
            # Step 4
            idct_yn = idct_2d_fp(y_n)
            s_n = u2 + (idct_yn - y_n)
            
            # Write to file
            f.write(f"BLOCK {b}\n")
            
            f.write("X_N:\n")
            for r in range(8):
                f.write(" ".join(map(str, x_n[r])) + "\n")
                
            f.write("S_PREV:\n")
            for r in range(8):
                f.write(" ".join(map(str, s_prev[r])) + "\n")
                
            f.write("Y_N:\n")
            for r in range(8):
                f.write(" ".join(map(str, y_n[r])) + "\n")
                
            f.write("S_N:\n")
            for r in range(8):
                f.write(" ".join(map(str, s_n[r])) + "\n")
                
            # Update s_prev for next block
            s_prev = s_n.copy()
            
    print(f"Generated testbench vectors at {out_path}")

if __name__ == "__main__":
    generate_vectors()
