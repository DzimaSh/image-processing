*Read this in other languages: [English](lab2_specification.md), [Беларуская](lab2_specification.be.md).*

---

# Lab 2 Technical Specification: 2D Non-Separable Wavelet Transforms

## 1. Introduction

In image processing, the standard 2D Discrete Wavelet Transform (DWT) is usually computed using a **separable** approach, applying 1D horizontal filtering across rows, followed by 1D vertical filtering down columns. While simple, this separable approach introduces significant processing latency (columns cannot be computed until all rows are finished), requires transposition buffers, and introduces multiple rounding steps that accumulate quantization noise in integer-to-integer coding.

This laboratory work implements **2D Non-Separable Wavelet Transforms**, which operate directly on 2D neighborhoods or multi-channel subgrids. We implement two architectures:

1. **2D Non-Separable Quaternionic Paraunitary Filter Bank (2-D NS Q-PUFB)** using a reversible, integer-to-integer block-lifting structure based on the research from BSUIR Minsk (*Rybenkov & Petrovsky*).
2. **2D Non-Separable CDF 9/7 Lifting Scheme**, which processes 2x2 image blocks in a single pass of direct 2D prediction and update steps, minimizing memory barriers and synchronization points (*David Bařina*).

---

## 2. Architecture 1: Quaternionic Multiplier Block-Lifting Transform (Q-PUFB)

The 2D Non-Separable Quaternionic Paraunitary Filter Bank (2-D NS Q-PUFB) operates on a 4-channel representation of the image. The 2D grid is split into non-overlapping $2 \times 2$ pixel blocks, represented as a 4D vector:

$$\mathbf{x} = \begin{bmatrix} x_{ee} \\ x_{eo} \\ x_{oe} \\ x_{oo} \end{bmatrix}$$

where:

* $x_{ee}[i, j] = x[2i, 2j]$ (Even-Even)
* $x_{eo}[i, j] = x[2i, 2j+1]$ (Even-Odd)
* $x_{oe}[i, j] = x[2i+1, 2j]$ (Odd-Even)
* $x_{oo}[i, j] = x[2i+1, 2j+1]$ (Odd-Odd)

### 2.1 Mathematical Formulation of Quaternionic Multiplication

The quaternionic multiplication operator $\mathbf{M}_+(Q)$ is a $4 \times 4$ real matrix parameterized by a unit-norm quaternion $Q = q_1 + q_2 i + q_3 j + q_4 k$ ($q_1^2 + q_2^2 + q_3^2 + q_4^2 = 1$):

$$\mathbf{M}_+(Q) = \begin{bmatrix} \mathbf{C}(Q) & -\mathbf{S}(Q) \\ \mathbf{S}(Q) & \mathbf{C}(Q) \end{bmatrix}$$

where the $2 \times 2$ submatrices are:

$$\mathbf{C}(Q) = \begin{bmatrix} q_1 & -q_2 \\ q_2 & q_1 \end{bmatrix}, \quad \mathbf{S}(Q) = \begin{bmatrix} q_3 & q_4 \\ q_4 & -q_3 \end{bmatrix}$$

### 2.2 Reversible Block-Lifting Factorization

To achieve a perfectly reversible, integer-to-integer transform (suitable for lossless image coding), $\mathbf{M}_+(Q)$ is factorized into a cascade of three block-lifting stages:

$$\mathbf{M}_+(Q) = \mathbf{U}(Q) \mathbf{L}(Q) \mathbf{V}(Q) = \begin{bmatrix} \mathbf{I}_2 & \mathbf{F}(Q) \\ \mathbf{0} & \mathbf{I}_2 \end{bmatrix} \begin{bmatrix} \mathbf{I}_2 & \mathbf{0} \\ \mathbf{G}(Q) & \mathbf{I}_2 \end{bmatrix} \begin{bmatrix} \mathbf{I}_2 & \mathbf{H}(Q) \\ \mathbf{0} & \mathbf{I}_2 \end{bmatrix}$$

where:

* $\mathbf{I}_2$ is the $2 \times 2$ identity matrix.
* The real-valued lifting matrices are:

  $$\mathbf{G}(Q) = \mathbf{S}(Q)$$
  $$\mathbf{F}(Q) = (\mathbf{C}(Q) - \mathbf{I}_2)\mathbf{S}(Q)^{-1}$$
  $$\mathbf{H}(Q) = \mathbf{S}(Q)^{-1}(\mathbf{C}(Q) - \mathbf{I}_2)$$

Given the algebraic properties of $\mathbf{S}(Q)$, its inverse is:

$$\mathbf{S}(Q)^{-1} = \frac{1}{q_3^2 + q_4^2} \mathbf{S}(Q)$$

This allows us to write $\mathbf{F}(Q)$ and $\mathbf{H}(Q)$ as trace-free, symmetric matrices:

$$\mathbf{F}(Q) = \begin{bmatrix} f_a & f_b \\ f_b & -f_a \end{bmatrix}, \quad \mathbf{H}(Q) = \begin{bmatrix} h_a & h_b \\ h_b & -h_a \end{bmatrix}$$

with:

$$f_a = \frac{(q_1 - 1)q_3 - q_2 q_4}{q_3^2 + q_4^2}, \quad f_b = \frac{(q_1 - 1)q_4 + q_2 q_3}{q_3^2 + q_4^2}$$
$$h_a = \frac{q_3(q_1 - 1) + q_4 q_2}{q_3^2 + q_4^2}, \quad h_b = \frac{q_4(q_1 - 1) - q_3 q_2}{q_3^2 + q_4^2}$$

### 2.3 Forward and Inverse Block-Lifting Steps

Let $\mathbf{v}_A = \begin{bmatrix} x_{ee} \\ x_{eo} \end{bmatrix}$ and $\mathbf{v}_B = \begin{bmatrix} x_{oe} \\ x_{oo} \end{bmatrix}$.

#### Forward Transform Steps

1. **Predict-like Block ($\mathbf{V}$):**
   $$\mathbf{v}_A \leftarrow \mathbf{v}_A + \text{round}(\mathbf{H}(Q) \cdot \mathbf{v}_B)$$
2. **Update-like Block ($\mathbf{L}$):**
   $$\mathbf{v}_B \leftarrow \mathbf{v}_B + \text{round}(\mathbf{G}(Q) \cdot \mathbf{v}_A)$$
3. **Predict-like Block ($\mathbf{U}$):**
   $$\mathbf{v}_A \leftarrow \mathbf{v}_A + \text{round}(\mathbf{F}(Q) \cdot \mathbf{v}_B)$$

#### Inverse Transform Steps

1. **Inverse $\mathbf{U}$:**
   $$\mathbf{v}_A \leftarrow \mathbf{v}_A - \text{round}(\mathbf{F}(Q) \cdot \mathbf{v}_B)$$
2. **Inverse $\mathbf{L}$:**
   $$\mathbf{v}_B \leftarrow \mathbf{v}_B - \text{round}(\mathbf{G}(Q) \cdot \mathbf{v}_A)$$
3. **Inverse $\mathbf{V}$:**
   $$\mathbf{v}_A \leftarrow \mathbf{v}_A - \text{round}(\mathbf{H}(Q) \cdot \mathbf{v}_B)$$

> [!NOTE]
> The rounding operator ensures that integer pixel values map to integers and reconstruct perfectly without any loss of precision, even when using fixed-point arithmetic!

---

## 3. Architecture 2: 2D Non-Separable CDF 9/7 Wavelet Transform

The 2D Non-Separable CDF 9/7 DWT operates directly on the $2 \times 2$ grid coordinates. In place of applying 1D horizontal steps and then 1D vertical steps, it combines these operations into unified 2D steps.

### 3.1 Interleaved 2D Lifting Scheme

The transform progresses through 4 non-separable 2D lifting steps:

1. **2D Predict Step 1 (Alpha):**
   Predicts high-frequency subbands from even-even subgrid:
   $$x_{eo}[i, j] \leftarrow x_{eo}[i, j] + \text{round}\left( \alpha \cdot (x_{ee}[i, j] + x_{ee}[i, j+1]) \right)$$
   $$x_{oe}[i, j] \leftarrow x_{oe}[i, j] + \text{round}\left( \alpha \cdot (x_{ee}[i, j] + x_{ee}[i+1, j]) \right)$$
   $$x_{oo}[i, j] \leftarrow x_{oo}[i, j] + \text{round}\left( \alpha \cdot (x_{eo}[i, j] + x_{eo}[i+1, j] + x_{oe}[i, j] + x_{oe}[i, j+1]) \right)$$

2. **2D Update Step 1 (Beta):**
   Updates smooth and high-frequency subgrids:
   $$x_{ee}[i, j] \leftarrow x_{ee}[i, j] + \text{round}\left( \beta \cdot (x_{eo}[i, j-1] + x_{eo}[i, j] + x_{oe}[i-1, j] + x_{oe}[i, j]) \right)$$

3. **2D Predict Step 2 (Gamma):**
   $$x_{eo}[i, j] \leftarrow x_{eo}[i, j] + \text{round}\left( \gamma \cdot (x_{ee}[i, j] + x_{ee}[i, j+1]) \right)$$
   $$x_{oe}[i, j] \leftarrow x_{oe}[i, j] + \text{round}\left( \gamma \cdot (x_{ee}[i, j] + x_{ee}[i+1, j]) \right)$$
   $$x_{oo}[i, j] \leftarrow x_{oo}[i, j] + \text{round}\left( \gamma \cdot (x_{eo}[i, j] + x_{eo}[i+1, j] + x_{oe}[i, j] + x_{oe}[i, j+1]) \right)$$

4. **2D Update Step 2 (Delta):**
   $$x_{ee}[i, j] \leftarrow x_{ee}[i, j] + \text{round}\left( \delta \cdot (x_{eo}[i, j-1] + x_{eo}[i, j] + x_{oe}[i-1, j] + x_{oe}[i, j]) \right)$$

5. **2D Scaling:**
   $$\begin{aligned}
   x_{ee} &\leftarrow K^2 \cdot x_{ee} \\
   x_{eo} &\leftarrow x_{eo} \\
   x_{oe} &\leftarrow x_{oe} \\
   x_{oo} &\leftarrow \frac{1}{K^2} \cdot x_{oo}
   \end{aligned}$$

---

## 4. Fixed-Point Optimization

All equations are evaluated using **Q16 fixed-point arithmetic** with round-to-nearest-even division:

$$\text{mul\_q16}(A, B) = \frac{A \cdot B + 32768}{65536}$$

This guarantees hardware compatibility and extremely fast execution times.
