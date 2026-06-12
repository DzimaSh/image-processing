*Read this in other languages: [English](lab3_specification.md), [Беларуская](lab3_specification.be.md).*

---

# Lab 3 Technical Specification: Lossless-to-Lossy (L2L) Block-Ladder Image Compression

## 1. Introduction

Traditional image compression systems (like JPEG) are inherently lossy because they apply quantization to DCT coefficients, which discards high-frequency information and introduces irreversible errors. To achieve lossless reconstruction, separate systems must be used, which typically have lower compression ratios.

This laboratory work implements a **Lossless-to-Lossy (L2L) Image Compression System** based on the **block-ladder (lifting) structural parametrization** of the 2D DCT-IDCT block transforms. The L2L framework combines lossy and lossless modes in a single unified processing pipeline:

1.  **Lossy Mode (Without SIB):** Compresses the image using standard block-based transformations. However, due to the presence of constant frequency leakage (DC-leakage or *DC leakages*) in the inverse DCT filters (i.e. loss of first-order regularity in all subband channels except the first), the reconstructed image displays a characteristic **checkerboard grid artifact** (chess pattern).
2.  **Lossless Mode (With SIB):** By capturing the cumulative rounding and quantization errors in an iterative state chain, we generate a final state block called the **Side Information Block (SIB)**. Transmitting this single block ($s_K$) as side metadata enables the decoder to perfectly reverse all intermediate rounding stages, yielding **perfect (lossless) reconstruction** (PSNR = $\infty$ dB, MSE = 0) with negligible data overhead.

---

## 2. Block-Ladder Parametrization (Lifting Scheme)

The L2L compression system represents the 2D DCT ($C_{2D}$) and 2D IDCT ($D_{2D}$) transforms through a 4-stage block-lifting structure. The image is decomposed into non-overlapping $M \times M$ blocks (here, $M=8$) in raster order.

Let:
*   $x_n$ be the $n$-th input spatial block.
*   $y_n$ be the $n$-th output transform coefficient block.
*   $s_n$ be the state block after processing block $n$ (with $s_{-1} = \mathbf{0}$).

The 2D block transforms are defined as:
*   $C_{2D}(X) = \mathbf{C} \cdot X \cdot \mathbf{C}^T$ (Forward 2D DCT)
*   $D_{2D}(X) = \mathbf{D} \cdot X \cdot \mathbf{D}^T$ (Inverse 2D IDCT, where $\mathbf{D} = \mathbf{C}^T$)

### 2.1 Forward Transform Steps (Encoder)

For each block $n = 0, 1, \dots, K-1$:

1.  **Stage 1:**
    $$u_1 = x_n - \text{round}\left( D_{2D}(s_{n-1}) \right)$$
2.  **Stage 2:**
    $$u_2 = s_{n-1} + \text{round}\left( C_{2D}(u_1) - u_1 \right)$$
3.  **Stage 3:**
    $$y_n = u_1 + u_2$$
4.  **Stage 4 (State Update):**
    $$s_n = u_2 + \text{round}\left( D_{2D}(y_n) - y_n \right)$$

### 2.2 Inverse Transform Steps (Decoder)

To decode, the state chain is evaluated backwards from $n = K-1$ down to $0$:

1.  **Inverse Stage 4:**
    $$u_2 = s_n - \text{round}\left( D_{2D}(y_n) - y_n \right)$$
2.  **Inverse Stage 3:**
    $$u_1 = y_n - u_2$$
3.  **Inverse Stage 2:**
    $$s_{n-1} = u_2 - \text{round}\left( C_{2D}(u_1) - u_1 \right)$$
4.  **Inverse Stage 1:**
    $$x_n = u_1 + \text{round}\left( D_{2D}(s_{n-1}) \right)$$

> [!IMPORTANT]
> *   In **Lossless Mode**, the decoder initializes the backward state chain with $s_{K-1} = s_K - \text{round}(D_{2D}(y_{K-1}) - y_{K-1})$, where $s_K$ is the transmitted SIB.
> *   In **Lossy Mode**, the SIB is ignored and the final state is initialized to zero ($s_K = \mathbf{0}$).

---

## 3. Fixed-Point Arithmetic Specification

To ensure exact reproducibility between the software model and hardware implementation, the arithmetic uses **12-bit fixed-point representation (Q1.10 format)**:

*   **Total Word Length:** 12 bits.
*   **Integer Parts:** 2 bits (1 sign bit, 1 integer bit).
*   **Fractional Parts:** 10 bits.
*   **Dynamic Range:** $[-2.0, 2.0)$ with a step size of $2^{-10} \approx 0.00097656$.
*   **Quantization Mode:** Convergent Rounding (Round-to-nearest-even / `AP_RND_CONV`).
*   **Overflow Mode:** Truncation (`AP_TRN`).

To prevent intermediate overflows during matrix multiplications, intermediate accumulations are performed using a **32-bit wider accumulator** before casting back to the 12-bit `coeff_t` format.

---

## 4. Hardware Architecture Design (Vivado HLS)

The hardware accelerator is implemented in synthesizable C++ Vivado HLS. The pipeline consists of the following core functional modules:

### 4.1 Interface Signals
The top-level synthesizable IP core [fwd_ladder_step_hls](file:///d:/Labs/master/image_processing/hardware/src/l2l_transform.cpp#L94) processes one block step:
*   `x_n`: Input spatial block array ($8 \times 8$).
*   `s_prev`: State input block array ($8 \times 8$).
*   `y_n`: Output transform coefficient block array ($8 \times 8$).
*   `s_n`: State output block array ($8 \times 8$).

### 4.2 HLS Optimization Pragmas
To achieve maximum performance and latency minimization, the following pragmas are applied:

1.  **Array Partitioning:**
    ```cpp
    #pragma HLS ARRAY_PARTITION variable=in complete dim=1
    ```
    This splits the block matrices along row dimensions, allowing parallel row read/write in a single clock cycle.
2.  **Loop Unrolling:**
    ```cpp
    #pragma HLS UNROLL
    ```
    Applied to the matrix multiplication dot-products, creating parallel multiplier arrays.
3.  **Pipelining:**
    ```cpp
    #pragma HLS PIPELINE II=1
    ```
    Applied to the outer matrix multiplication loops to schedule iterations overlappingly with an Initiation Interval of 1.
