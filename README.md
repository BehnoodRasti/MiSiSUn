# MiSiSUn

**MiSiSUn: Minimum Simplex Semisupervised Unmixing**

---

## Introduction

MiSiSUn is a semi-supervised hyperspectral unmixing method with an open-source Python/PyTorch package. In addition to the proposed MiSiSUn method, the package includes eight competitive baseline algorithms.

MiSiSUn solves the following nonconvex optimization problem:

```math
\begin{aligned}
(\hat{\mathbf{B}}, \hat{\mathbf{A}})
&= \arg\min_{\mathbf{B}, \mathbf{A}}
\frac{1}{2}\left\| \mathbf{Y} - \mathbf{D}\mathbf{B}\mathbf{A} \right\|_F^2 
\quad + \lambda \left\| \mathbf{D}\mathbf{B} - \mathbf{m}\mathbf{1}_r^T \right\| \\
\text{s.t.}\quad
&\mathbf{B} \ge 0,\ \mathbf{1}_m^T \mathbf{B} = \mathbf{1}_r^T, \\
&\mathbf{A} \ge 0,\ \mathbf{1}_r^T \mathbf{A} = \mathbf{1}_n^T.
\end{aligned}
```

where \(\mathbf{m}\) contains the mean values of the spectral pixels, i.e.,

```math
\mathbf{m} = \frac{1}{n}\mathbf{Y}\mathbf{1}_n.
```

This regularization term pulls the endmembers toward the center of mass.

> **Note**
> The provided tools can also be used for signal and image processing applications beyond hyperspectral unmixing, such as source separation.

## MiSiSUn Features

- Semi-supervised unmixing framework (the dictionary \(\mathbf{D}\) must be provided)
- 9 unmixing methods: `MiSiSUn`, `FaSUn`, `SUnS`, `SUnAA`, `SUnCNN`, `S2WSU`, `MUA_SLIC`, `CLSUnSAL`, `SUnSAL`
- 2 evaluation metrics: `SRE`, `RMSE`
- 4 simulated datasets located under `./data/`

## License

MiSiSUn is distributed under the MIT License.

## Citing MiSiSUn

If you use MiSiSUn in your work, please cite:

> B. Rasti, B. Koirala, and P. Scheunders, “MiSiSUn: Minimum Simplex Semisupervised Unmixing,” arXiv preprint arXiv:2603.20263, Mar. 13, 2026, doi: 10.48550/arXiv.2603.20263.

## Installation

### Using `conda`

We recommend using a `conda` environment to install MiSiSUn.

In the following steps, we use `conda` to manage the Python environment and `pip` to install the required packages. If you do not have `conda`, install it via [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda create --name MiSiSUn python=3.10
```

Activate the environment:

```bash
conda activate MiSiSUn
```

Clone the GitHub repository:

```bash
git clone git@github.com:BehnoodRasti/MiSiSUn.git
```

Move into the project directory and install the required packages:

```bash
cd MiSiSUn
pip install -r requirements.txt
```

## Important Note

PyTorch is not included in `requirements.txt`. You need to install PyTorch separately according to your OS and CUDA version. See the [official installation guide](https://pytorch.org/get-started/locally/).

The package has been tested on Linux and Windows using Python 3.10 and `pytorch-cuda=11.8`.

## Getting Started

To run an experiment, you need to define a few parameters:

- `data`: hyperspectral unmixing dataset (`sim1`, `rho70`, `rho85`, `rho100`)
- `model`: unmixing model (e.g. `MiSiSUn`)
- `SNR`: input SNR (*optional*)

An example command is:

```bash
python unmixing.py hsi=rho100 noise=20dB model=MiSiSUn model.T=10000 model.TA=5 model.TB=5 model.mu1=50.0 model.mu2=2.0 model.mu3=1.0 model.lamb=0.001 noise.seed=0,1,2,3,4 --multirun
```

## Data

### Data Format

Each dataset consists of a dedicated `.mat` file containing the following keys:

- `Y`: original hyperspectral image (dimension `p x n`)
- `D`: endmember library (dimension `p x m`)
- `A`: ground-truth abundances (dimension `r x n`)
- `h`: number of image rows
- `w`: number of image columns
- `r`: number of endmembers
- `p`: number of spectral channels
- `n`: number of pixels (`n = h * w`)
- `m`: number of atoms in the library

## Parameter Tuning

### Fine Tuning

You may need to fine-tune the model parameters for your application. For example, you can change the number of iterations for the outer loop (`T`) there.

## Visualization

You can use `visualization.py` to visualize abundance maps and estimated endmembers (for AA-based models only).

```bash
python visualize.py --path results.mat --mode both --topk 12 --endmember-topk 12
```
