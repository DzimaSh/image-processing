# Advanced Image Processing & Wavelet Transforms

[![Python Application CI](https://github.com/DzimaSh/image-processing/actions/workflows/python-app.yml/badge.svg)](https://github.com/DzimaSh/image-processing/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-green.svg)](https://www.python.org/)

Welcome to the **Advanced Image Processing & Wavelet Transforms** university/research repository. This codebase is designed to host, test, and document professional, high-performance image processing modules.

## 📚 Repository Lab Modules

* **Lab 1: CDF 9/7 Wavelet Transform via Separable Lifting Scheme**
  A production-ready, NumPy-vectorized implementation of the **Cohen-Daubechies-Feauveau (CDF) 9/7 Biorthogonal Wavelet Transform** utilizing the separable Lifting Scheme, mapped entirely to **Q16 Fixed-Point Arithmetic** with rounded integer scaling.
* **Lab 2: 2D Non-Separable Wavelet Transforms & Paraunitary Filter Banks**
  An advanced implementation of the **Reversible 2D Non-Separable Quaternionic Paraunitary Filter Bank (Q-PUFB)** using a block-lifting structure (*Rybenkov & Petrovsky*), and the **2D Non-Separable CDF 9/7 Wavelet Transform** using direct 2D lifting steps (*David Bařina*), implemented with Q16 fixed-point precision.

---

## 🚀 Key Features

* **Multi-Lab Architecture:** Features a clean, modular structure where Lab 1 and Lab 2 reside in their own packages while sharing a standard `shared/` folder for core image I/O and statistical evaluation metrics.
* **Fixed-Point Arithmetic (Q16 Format):** Employs scaling by $2^{16} = 65536$ and round-to-nearest scaling logic `(A * B + 32768) >> 16` to maintain floating-point-like accuracy with pure integer arithmetic.
* **Vectorized Lifting Schemes:** Implements lifting steps optimized with NumPy for extreme execution speed and perfect numerical reconstruction.
* **Symmetric Boundary Handling:** Resolves edge conditions natively using half-sample symmetric extensions, preventing visual borders/artifacts.
* **Image Processing CLI Pipelines:** Dedicated CLI applications to process images, generate DWT visual representations, reconstruct images, and output metrics.
* **Statistical Metrics:** Computes Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) to measure the exact reconstruction fidelity.
* **CI/CD Integrated:** Configured with GitHub Actions for immediate formatting/lint checks via Ruff and unit testing coverage via Pytest.

---

## 📂 Repository Structure

The codebase is organized into a clean, multi-lab architecture to support separate assignments with shared core libraries:

```text
image_processing/
│
├── .github/
│   └── workflows/
│       └── python-app.yml      # CI/CD for Ruff linting and Pytest
├── data/
│   ├── .gitkeep                # Calibration & sample images
│   └── output/                 # Saved output visualizations and reconstructed images
├── docs/
│   ├── lab1_specification.md   # Mathematical specification & LaTeX formulas (Lab 1)
│   └── lab2_specification.md   # Mathematical specification & LaTeX formulas (Lab 2)
├── src/
│   ├── __init__.py
│   ├── shared/                 # Shared resources (Metrics, Image I/O)
│   │   ├── __init__.py
│   │   ├── metrics.py          # Quality metrics: MSE and PSNR
│   │   └── image_io.py         # Image readers/writers and subband visualizations
│   ├── lab1/                   # Laboratory Work 1
│   │   ├── __init__.py
│   │   └── lifting_scheme.py   # Vectorized separable DWT/IDWT logic
│   └── lab2/                   # Laboratory Work 2
│       ├── __init__.py
│       └── algorithm.py        # 2D non-separable & Quaternionic filter bank
├── tests/
│   ├── __init__.py
│   ├── test_lab1.py            # Unit tests for Lab 1 (separable)
│   ├── test_lab2.py            # Unit tests for Lab 2 (non-separable)
│   └── test_metrics.py         # Mathematical tests for PSNR/MSE metrics
├── scripts/
│   ├── run_lab1.py             # CLI runner for Lab 1
│   └── run_lab2.py             # CLI runner for Lab 2
├── .gitignore                  # Robust ignore definitions
├── requirements.txt            # Python dependencies
├── README.md                   # Repository documentation
└── LICENSE                     # MIT License
```

---

## 📖 Mathematical Foundations

* **Lab 1:** The lifting scheme factorizes the polyphase matrix of the CDF 9/7 biorthogonal filters into a sequence of prediction and update steps, facilitating direct in-place computations. For details, refer to the [Lab 1 Specification](docs/lab1_specification.md).
* **Lab 2:** Implements a multi-channel 2D non-separable paraunitary filter bank based on hypercomplex quaternionic algebra, factorized into three block-lifting stages ($U, L, V$) for lossless integer-to-integer mapping, and a 2D non-separable CDF 9/7 wavelet transform. For details, refer to the [Lab 2 Specification](docs/lab2_specification.md).

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/DzimaSh/image-processing.git
   cd image-processing
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Running Lab 1 (Separable CDF 9/7 DWT)

The CLI script `scripts/run_lab1.py` executes Lab 1:

```bash
# Run on automatically generated synthetic image
python scripts/run_lab1.py

# Run on custom image with 3 decomposition levels
python scripts/run_lab1.py --image data/my_image.png --levels 3
```

### 2. Running Lab 2 (2D Non-Separable Transforms)

The CLI script `scripts/run_lab2.py` executes Lab 2. You can choose between the quaternionic filter bank or the 2D non-separable CDF 9/7 transform:

```bash
# Run Quaternionic Paraunitary Filter Bank (perfect integer-to-integer reconstruction)
python scripts/run_lab2.py --transform quaternionic

# Run 2D Non-Separable CDF 9/7 Wavelet Transform (high-fidelity reconstruction)
python scripts/run_lab2.py --transform non-separable-cdf

# Run on your own grayscale image
python scripts/run_lab2.py --image data/my_image.png --transform quaternionic
```

### Output Visualizations (Stored in `data/output/`)

* `lab2_vis_quaternionic.png` & `lab2_vis_non-separable-cdf.png`: Visual representation of transform coefficients.
* `lab2_recon_quaternionic.png` & `lab2_recon_non-separable-cdf.png`: Reconstructed images from transform coefficients.

---

## 🧪 Testing

We maintain high test coverage for mathematical correctness and perfect reversibility. Run the test suite using `pytest`:

```bash
# Run all tests (Lab 1, Lab 2, and metrics)
pytest -v

# Run specific lab tests
pytest tests/test_lab1.py -v
pytest tests/test_lab2.py -v
```

All tests are verified automatically via GitHub Actions upon push and pull requests.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
