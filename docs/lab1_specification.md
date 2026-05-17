# Lab 1 Technical Specification: CDF 9/7 DWT via Fixed-Point Lifting Scheme

## 1. Introduction

Wavelet transforms are powerful tools for multiresolution signal analysis, widely used in image compression standards such as JPEG 2000. This document specifies the implementation of a **1D and 2D Discrete Wavelet Transform (DWT)** and **Inverse Discrete Wavelet Transform (IDWT)** using the biorthogonal **Cohen-Daubechies-Feauveau (CDF) 9/7 filter bank**. 

To optimize performance on embedded systems or hardware lacking floating-point units (FPUs), the entire algorithm is implemented using **Fixed-Point Arithmetic** under a **Lifting Scheme** architecture.

---

## 2. The Lifting Scheme Architecture

The Lifting Scheme is a method to construct biorthogonal wavelets by factorizing the wavelet filter bank into a sequence of prediction and update steps (primal and dual lifting steps). This structure provides several advantages over classical convolution-based DWT:

1. **In-place computation:** Coefficients can be updated directly in the input array.
2. **Computational efficiency:** Reduces the number of multiplications and additions by up to $50\%$.
3. **Easy reversibility:** The inverse transform is structurally symmetric to the forward transform; reversing the steps and changing the signs of the coefficients restores the original signal.

### 2.1 Forward 1D CDF 9/7 Lifting Step Equations

Let $x[n]$ be a 1D discrete input signal of length $N$. The signal is split into even-indexed samples $x_e[n] = x[2n]$ and odd-indexed samples $x_o[n] = x[2n+1]$ for $n = 0, 1, \dots, \frac{N}{2} - 1$.

The forward lifting scheme progresses through four sequential phases followed by scaling:

1. **Predict Phase 1 ($\alpha$):**
   $$d^{[1]}[n] = x_o[n] + \alpha \left( x_e[n] + x_e[n+1] \right)$$
   *Predicts odd samples from adjacent even samples and computes detail coefficients.*

2. **Update Phase 1 ($\beta$):**
   $$s^{[1]}[n] = x_e[n] + \beta \left( d^{[1]}[n-1] + d^{[1]}[n] \right)$$
   *Updates even samples using adjacent detail coefficients to preserve the running mean.*

3. **Predict Phase 2 ($\gamma$):**
   $$d^{[2]}[n] = d^{[1]}[n] + \gamma \left( s^{[1]}[n] + s^{[1]}[n+1] \right)$$
   *Refines details using the updated smooth coefficients.*

4. **Update Phase 2 ($\delta$):**
   $$s^{[2]}[n] = s^{[1]}[n] + \delta \left( d^{[2]}[n-1] + d^{[2]}[n] \right)$$
   *Refines smooth coefficients to maintain smoothness.*

5. **Scaling ($K$):**
   $$s[n] = K \cdot s^{[2]}[n]$$
   $$d[n] = \frac{1}{K} \cdot d^{[2]}[n]$$
   *Final normalization step to obtain the low-frequency approximation $s$ and high-frequency details $d$.*

#### Floating-Point Lifting Coefficients:
$$\begin{aligned}
\alpha &\approx -1.586134342059924 \\
\beta &\approx -0.052980118572961 \\
\gamma &\approx 0.882911075530934 \\
\delta &\approx 0.443506852043971 \\
K &\approx 1.230174104914001 \\
\frac{1}{K} &\approx 0.812893066115961
\end{aligned}$$

---

### 2.2 Inverse 1D CDF 9/7 Lifting Step Equations

Reconstruction is achieved by reversing the operations of the forward transform in exact reverse order with inverted signs:

1. **Inverse Scaling:**
   $$s^{[2]}[n] = \frac{1}{K} \cdot s[n]$$
   $$d^{[2]}[n] = K \cdot d[n]$$

2. **Inverse Update Phase 2 ($\delta$):**
   $$s^{[1]}[n] = s^{[2]}[n] - \delta \left( d^{[2]}[n-1] + d^{[2]}[n] \right)$$

3. **Inverse Predict Phase 2 ($\gamma$):**
   $$d^{[1]}[n] = d^{[2]}[n] - \gamma \left( s^{[1]}[n] + s^{[1]}[n+1] \right)$$

4. **Inverse Update Phase 1 ($\beta$):**
   $$x_e[n] = s^{[1]}[n] - \beta \left( d^{[1]}[n-1] + d^{[1]}[n] \right)$$

5. **Inverse Predict Phase 1 ($\alpha$):**
   $$x_o[n] = d^{[1]}[n] - \alpha \left( x_e[n] + x_e[n+1] \right)$$

6. **Recombination:**
   Combine $x_e[n]$ and $x_o[n]$ back into the reconstructed signal $x[n]$.

---

## 3. Fixed-Point Arithmetic Design (Q16 Format)

To achieve fixed-point representation, we scale real values by a factor of $2^{16} = 65536$ (known as Q16 format, or $Q_{15.16}$ for signed 32-bit integers). 

### 3.1 Scaling Lifting Coefficients

Let $S = 65536$ be the scaling factor. The Q16 representation of each coefficient is computed via:
$$C_{\text{fix}} = \text{round}(C \times S)$$

$$\begin{aligned}
\alpha_{\text{fix}} &= \text{round}(-1.586134342059924 \times 65536) = -103948 \\
\beta_{\text{fix}} &= \text{round}(-0.052980118572961 \times 65536) = -3472 \\
\gamma_{\text{fix}} &= \text{round}(0.882911075530934 \times 65536) = 57863 \\
\delta_{\text{fix}} &= \text{round}(0.443506852043971 \times 65536) = 29066 \\
K_{\text{fix}} &= \text{round}(1.230174104914001 \times 65536) = 80620 \\
(1/K)_{\text{fix}} &= \text{round}(0.812893066115961 \times 65536) = 53273
\end{aligned}$$

### 3.2 Fixed-Point Multiplication and Rounding

When multiplying two Q16 numbers, the raw result is in Q32 format. We must scale it back to Q16 by right-shifting by 16 bits.
$$\text{Product}_{\text{Q16}} = (A_{\text{Q16}} \times B_{\text{Q16}}) \gg 16$$

To avoid systematic negative bias from floor division (`>>`), we apply **round-to-nearest-even** or standard rounding by adding half the denominator ($2^{15} = 32768$) before shifting:
$$\text{Product}_{\text{Q16\_rounded}} = (A_{\text{Q16}} \times B_{\text{Q16}} + 32768) \gg 16$$

---

## 4. Boundary Extensions

At boundaries (indices $n < 0$ or $n \ge \frac{N}{2}$), values are extended symmetrically to prevent artifacts and energy leakages:

* **Left Boundary:** $x[-1] = x[1]$
* **Right Boundary:** $x[M] = x[M-2]$ where $M = \frac{N}{2}$ is the boundary limit.

---

## 5. Metrics Formulation

We evaluate the reconstruction quality using the following statistical measures:

### 5.1 Mean Squared Error (MSE)

Given the original image $I$ and the reconstructed image $\hat{I}$, both of dimensions $H \times W$:
$$\text{MSE} = \frac{1}{H \times W} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \left( I[i, j] - \hat{I}[i, j] \right)^2$$

### 5.2 Peak Signal-to-Noise Ratio (PSNR)

PSNR evaluates the ratio between the maximum possible power of a signal and the corrupting noise that affects its representation. For 8-bit grayscale images (where peak value $MAX_I = 255$):
$$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{255^2}{\text{MSE}} \right) \quad (\text{dB})$$

If $\text{MSE} = 0$, the reconstructed image is identical to the original; the PSNR is mathematically infinite ($\infty$).
