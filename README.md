# MiSiSUn
MiSiSUn: Minimum Simplex Semisupervised Unmixing

---

## Introduction

MiSiSUn is a method for Semi-supervised hyperspectral unmixing, which comes with an open-source Python/PyTorch Package. In addition to the novel proposed method (i.e., MiSiSUn), the package contains eight competitive algorithms. MiSiSUn solves the following nonconvex optimizations:

MiSiSUn (Fast Semisupervised Unmixing):
```math
  (\hat{\bf B},\hat{\bf A})=\arg\min_{{\bf B,A}} \frac{1}{2} || {\bf Y}-{\bf DBA}||_{F}^{2} + \lambda||{\bf DB} - {\bf m}{\bf 1}_{r}^{T}||~~~
{\rm s.t.}~~~{\bf B}\geq 0,{\bf 1}_{m}^{T}{\bf B}={\bf 1}_{r}^{T},  {\rm and } ~~~ {\bf A}\geq 0,{\bf 1}_{r}^{T}{\bf A}={\bf 1}_{n}^{T}.
```
where, ${\bf m}$ contains the mean values of the spectral pixels, i.e., ${\bf m}=\frac{1}{n}{\bf Y}{\bf 1}_n$.  This term pulls the endmembers toward the center of mass. 
```
Note: The provided tools can be used for signal and image processing applications beyond unmixing  such as source separation. 

## MiSiSUn Features

* Semisupervised category (Dictionary ${\bf D}$ should be provided)
* 9 unmixing methods (MiSiSUn, FaSUn, SUnS, SUnAA, SUnCNN, S2WSU, MUA_SLIC, CLSUnSAL, SUnSAL)
* 2 metrics (SRE, RMSE)
* 2 simulated datasets (located under `./data/`)

## License

MiSiSUn is distributed under MIT license.

## Citing MiSiSUn

B. Rasti, B. Koirala, and P. Scheunders, “MiSiSUn: Minimum Simplex Semisupervised Unmixing,” arXiv preprint arXiv:2603.20263, Mar. 13, 2026, doi: 10.48550/arXiv.2603.20263.

## Installation

### Using `conda`

We recommend using a `conda` virtual Python environment to install MiSiSUn.

In the following steps we will use `conda` to handle the Python distribution and `pip` to install the required Python packages.
If you do not have `conda`, please install it using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```
conda create --name MiSiSUn python=3.10
```

Activate the new `conda` environment to install the Python packages.

```
conda activate MiSiSUn
```

Clone the Github repository.

```
git clone git@github.com:BehnoodRasti/MiSiSUn.git
```

Change directory and install the required Python packages.

```
cd MiSiSUn && pip install -r requirements.txt
```

## Important Note

The PyTorch was not included in the requirements.txt. You'll need to separately install PyTorch according to your OS and CUDA, please take a look https://pytorch.org/get-started/locally/. We tested the package on both Linux and Windows using Python 3.10 and pytorch-cuda=11.8.


## Getting started

There are a few required parameters to define in order to run an experiment:

* `data`: hyperspectral unmixing dataset (DC1, DC2, MR70, MR85, and MR100)
* `model`: unmixing model (e.g., MiSiSUn)
* `SNR`: input SNR (*optional*)

An example of a corresponding command line is simply:

```shell
python unmixing.py hsi=rho100 noise=20dB model=MiSiSUn model.T=10000 model.TA=5 model.TB=5 model.mu1=50.0 model.mu2=2.0 model.mu3=1.0 model.lamb=0.001 noise.seed=0,1,2,3,4 --multirun
```

## Data

### Data format

Datasets consist in a dedicated `.mat` file containing the following keys:

* `Y`: original hyperspectral image (dimension `p` x `n`)
* `D`: endmembers library (dimension `p` x `m`)
* `A`: ground truth abundances (dimension `r` x `n`)
* `h`: HSI number of rows
* `w`: HSI number of columns
* `r`: number of endmembers
* `p`: number of channels
* `n`: number of pixels (`n` == `h`*`w`)
* `m`: number of atoms

## Parameter Tuning

### Fine Tuning

You may need to fine-tune the models' parameters for your application. For instance, for FaSUn, the parameters are indicated in config/model/FaSUn.yaml, and we can change the number of iterations for the outer loop (T) with the following line. 

## Visualization 

You can use visualization.py to visualize the abundance maps and estimated endmembers (only for AA-based models). 

```shell
python visualize.py --path results.mat --mode both --topk 12 --endmember-topk 12
```
