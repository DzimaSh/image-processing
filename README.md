# Fixed-Point CDF 9/7 Wavelet Transform via Lifting Scheme

[![Python Application CI](https://github.com/your-username/image_processing/actions/workflows/python-app.yml/badge.svg)](https://github.com/your-username/image_processing/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-green.svg)](https://www.python.org/)

A highly optimized, production-ready Python implementation of the **Cohen-Daubechies-Feauveau (CDF) 9/7 Biorthogonal Wavelet Transform** using the **Lifting Scheme** mapped entirely to **Q16 Fixed-Point Arithmetic**.

This repository corresponds to **Lab 1** of the Image Processing university/research curriculum. It demonstrates mathematically exact, in-place wavelet decompositions designed for systems without floating-point units (FPUs) or embedded hardware acceleration.

---

## 🚀 Key Features

* **Fixed-Point Arithmetic (Q16 Format):** Employs scaling by $2^{16} = 65536$ and round-to-nearest scaling logic `(A * B + 32768) >> 16` to maintain floating-point-like accuracy with pure integer arithmetic.
* **Vectorized Lifting Scheme:** Implements Predict 1, Update 1, Predict 2, Update 2, and scaling steps optimized with NumPy for extreme execution speed.
* **Symmetric Boundary Handling:** Resolves edge conditions natively using half-sample symmetric extensions, preventing visual borders/artifacts.
* **Image Processing CLI Pipeline:** CLI application to process images, perform multi-level decompositions, generate DWT visual representations, reconstruct images, and output metrics.
* **Statistical Metrics:** Computes Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) to measure the exact reconstruction fidelity.
* **CI/CD Integrated:** Configured with GitHub Actions for immediate formatting/lint checks via Ruff and unit testing coverage via Pytest.

---

## 📂 Repository Structure

The codebase is organized into modular packages to isolate the core mathematics from interface execution:

```text
project-root/
│
├── .github/
│   └── workflows/
│       └── python-app.yml      # CI/CD for Ruff linting and Pytest
├── data/
│   ├── .gitkeep                # Calibration & sample images
│   └── output/                 # Saved output visualizations and reconstructed images
├── docs/
│   └── lab1_specification.md   # Mathematical specification & LaTeX formulas
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── lifting_scheme.py   # Vectorized DWT and IDWT fixed-point logic
│   │   └── metrics.py          # Quality metrics: MSE, RMSE, PSNR
│   └── utils/
│       ├── __init__.py
│       └── image_io.py         # Image readers/writers and subband visualizations
├── tests/
│   ├── __init__.py
│   ├── test_lifting_scheme.py  # Unit tests verifying roundtrip reversibility
│   └── test_metrics.py         # Mathematical tests for PSNR/MSE metrics
├── scripts/
│   └── run_lab1.py             # Main CLI execution runner
├── .gitignore                  # Robust ignore definitions (excludes binary docx/pdfs)
├── requirements.txt            # Python dependencies (NumPy, OpenCV, Pytest, Ruff)
├── README.md                   # Repository documentation
└── LICENSE                     # MIT License
```

---

## 📖 Mathematical Foundation

The lifting scheme factorizes the polyphase matrix of the CDF 9/7 biorthogonal filters into a sequence of prediction and update steps, facilitating direct in-place computations.

For details on the specific lifting equations ($\alpha, \beta, \gamma, \delta$), the scaling parameter ($K$), the boundary reflection logic, and the Q16 scaling factors, please refer to the detailed [Technical Specification](docs/lab1_specification.md).

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/image_processing.git
   cd image_processing
   ```

2. **Create a Virtual Environment (Optional but recommended):**
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

The CLI script `scripts/run_lab1.py` acts as the primary execution engine. 

### 1. Run on a Synthetic Calibration Image (No file needed)
If you run the script without any arguments, it will automatically generate a custom high-frequency checkerboard calibration image in `data/sample_synthetic.png` and run the entire pipeline:
```bash
python scripts/run_lab1.py
```

### 2. Run on Your Own Grayscale/Color Image
```bash
python scripts/run_lab1.py --image data/my_image.png
```

### 3. Run with Multi-Level Wavelet Decomposition (e.g., 3 levels)
```bash
python scripts/run_lab1.py --image data/my_image.png --levels 3
```

### Output Files (Stored in `data/output/`):
* `reconstructed.png`: The fully reconstructed image from the wavelet coefficients.
* `dwt_vis_level_N.png`: A stunning, grid-aligned subband visualization highlighting LL (approximation), LH (horizontal details), HL (vertical details), and HH (diagonal details) bands.

---

## 🧪 Testing

We maintain a high test coverage for mathematical correctness and implementation safety. Run tests using `pytest`:

```bash
# Run all tests
pytest -v

# Run only lifting scheme roundtrip tests
pytest tests/test_lifting_scheme.py -v
```

All tests are verified and checked automatically via GitHub Actions upon push and pull requests.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
