*Read this in other languages: [English](lab3_specification.md), [Беларуская](lab3_specification.be.md).*

---

# Technical Specification: Lossless-to-Lossy (L2L) Block-Ladder Image Compression

## 1. Theoretical Background

### 1.1 Lossless-to-Lossy (L2L) Framework
Traditional image coding systems separate lossy compression (e.g., JPEG, which uses DCT with quantization) and lossless compression (e.g., JPEG-LS, PNG, which use predictive schemes). 
The Lossless-to-Lossy (L2L) framework unified under a single structural parametrization allows progressive decoding from lossy representation up to bit-true (perfect) reconstruction from the same compressed bitstream.

### 1.2 First-Order Regularity & DC Leakage (Checkerboard Effect)
The checkerboard pattern (grid artifact) is a well-known distortion in block-based transform systems working in lossy modes. 

A filter bank (or transform kernel) satisfies the **first-order regularity condition** if the frequency responses of all subband channels, except the first (DC) channel, have a zero at zero frequency:
$$H_k(\omega)\Big|_{\omega=0} = 0 \quad \text{for } k = 1, 2, \dots, M-1$$

In standard 2D DCT-IDCT block transforms:
*   The forward DCT matrix $\mathbf{C}$ concentrates the block's average energy into the DC coefficient $y[0,0]$.
*   The synthesis filters (columns of the IDCT matrix $\mathbf{D}$) are orthogonal and represent AC frequencies that integrate to zero.

However, when implementing these transforms in **finite-precision arithmetic** (fixed-point) inside a block-ladder lifting structure, rounding errors are introduced at each lifting step. Because the rounding operations are non-linear, they generate rounding noise. This noise contains a constant (DC) component that leaks into the AC subbands. 

Because the subband filters lose their first-order regularity in this parametrical ladder representation, this leaked DC energy manifests as a structured **checkerboard grid artifact** (chess pattern) in the spatial domain when the decoder reconstructs the image.

### 1.3 Side Information Block (SIB) Stabilization
To eliminate this leakage and achieve perfect reconstruction, a **Side Information Block (SIB)** is employed. 

The forward encoder chains the $8 \times 8$ image blocks in a 1D sequence. The rounding errors from each block $n$ are accumulated in the state block $s_n$:
$$s_n = u_2 + \text{round}\left( D_{2D}(y_n) - y_n \right)$$
where $s_{-1} = \mathbf{0}$.

At the end of the image (after $K$ blocks), the final state block $s_K$ (the SIB) contains the accumulated error history. 
*   In **Lossless Mode**, this single block $s_K$ is transmitted. The decoder uses it to run the state chain backward. This reverses every rounding step exactly, recovering the original pixels bit-for-bit (PSNR = $\infty$ dB, MSE = 0).
*   In **Lossy Mode**, the SIB is omitted ($s_K = \mathbf{0}$). This results in normal lossy reconstruction displaying the checkerboard pattern, allowing comparative quality assessments.

---

## 2. Mathematical Formulation

Let the 2D DCT and 2D IDCT operations be:
*   $C_{2D}(X) = \mathbf{C} \cdot X \cdot \mathbf{C}^T$
*   $D_{2D}(X) = \mathbf{D} \cdot X \cdot \mathbf{D}^T$ (where $\mathbf{D} = \mathbf{C}^T$)

### 2.1 Forward block-ladder steps (Encoder)
For each block $n = 0, 1, \dots, K-1$:
1.  $$u_1 = x_n - \text{round}\left( D_{2D}(s_{n-1}) \right)$$
2.  $$u_2 = s_{n-1} + \text{round}\left( C_{2D}(u_1) - u_1 \right)$$
3.  $$y_n = u_1 + u_2$$
4.  $$s_n = u_2 + \text{round}\left( D_{2D}(y_n) - y_n \right)$$

### 2.2 Inverse block-ladder steps (Decoder)
For each block $n = K-1, \dots, 0$:
1.  $$u_2 = s_n - \text{round}\left( D_{2D}(y_n) - y_n \right)$$
2.  $$u_1 = y_n - u_2$$
3.  $$s_{n-1} = u_2 - \text{round}\left( C_{2D}(u_1) - u_1 \right)$$
4.  $$x_n = u_1 + \text{round}\left( D_{2D}(s_{n-1}) \right)$$

---

## 3. Fixed-Point Specification

To match the hardware exactly, the software modeling and synthesizable hardware use **12-bit fixed-point arithmetic (Q1.10 format)**:
*   **Word Length:** 12 bits (1 sign bit, 1 integer bit, 10 fractional bits).
*   **Rounding Mode:** Convergent Rounding (Round-to-nearest-even / `AP_RND_CONV`).
*   **Overflow Mode:** Truncation (`AP_TRN`).
*   **Accumulator:** 32-bit wider variables are used during matrix multiplication summation to prevent intermediate overflow.

---

## 4. How to Launch the Application

### 4.1 Python Algorithmic Model
The Python implementation includes a CLI runner and a unit test suite.

#### Prerequisites
Ensure the virtual environment is active and dependencies are installed:
```bash
pip install -r requirements.txt
```

#### Run CLI Evaluation
The CLI script runs the L2L system on a grayscale image, compares Lossless (with SIB) vs Lossy (without SIB) modes, prints quality metrics, and saves the output images in `data/output/`.

1.  **Run with standard matrix DCT on synthetic calibration image:**
    ```bash
    python scripts/run_lab3.py
    ```
2.  **Run with standard matrix DCT on a custom image:**
    ```bash
    python scripts/run_lab3.py --image path/to/your_image.png
    ```
3.  **Run using the fast Loeffler DCT kernel:**
    ```bash
    python scripts/run_lab3.py --use-loeffler
    ```

#### Run Unit Tests
To verify transform invertibility and perfect reconstruction properties:
```bash
pytest tests/test_lab3.py -v
```

---

### 4.2 AMD Vivado / Vitis HLS Hardware Design
The hardware architecture is verified via a C-simulation testbench using golden vectors.

#### 1. Generate Testbench Vectors
Run the generator script to create bit-accurate inputs/outputs (using scaled Q1.10 values to avoid `ap_fixed<12, 2>` overflow):
```bash
python scripts/generate_tb_vectors.py
```
This writes the vectors to [tb_vectors.txt](file:///d:/Labs/master/image_processing/hardware/src/tb_vectors.txt).

#### 2. Execute HLS Synthesis & Verification
Open the Vivado HLS Command Prompt (or Vitis HLS console) and run the Tcl script:
```bash
vivado_hls -f hardware/scripts/hls_script.tcl
```
or in Vitis HLS:
```bash
vitis_hls -f hardware/scripts/hls_script.tcl
```

This script will automatically:
1.  Initialize the project `l2l_hls_project`.
2.  Add synthesizable source files and testbenches.
3.  **C Simulation (`csim_design`):** Compiles and runs the C++ testbench in [l2l_tb.cpp](file:///d:/Labs/master/image_processing/hardware/src/l2l_tb.cpp), verifying the design against Python golden vectors.
4.  **C Synthesis (`csynth_design`):** Generates RTL Verilog/VHDL code and scheduling reports.
5.  **C/RTL Co-Simulation (`cosim_design`):** Verifies the synthesized RTL block inside a simulator using the C++ testbench.
6.  **Export IP (`export_design`):** Packages the synthesized block as a Vivado IP Core.
