import logging
import argparse
from pathlib import Path

import scipy.io as sio
import matplotlib.pyplot as plt


# CUPRITE DIMENSIONS
H = 250
W = 191

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", "-p", type=str)

    args = parser.parse_args()

    result_path = Path(args.path)
    assert result_path.is_file()

    result = sio.loadmat(result_path)
    # A_hat shape => (r, n)
    # where 'r' denotes the number of actual endmembers
    # where 'n' denotes the number of pixels
    abundances = result["A_hat"]

    r = len(abundances)

    fig, ax = plt.subplots(ncols=r, nrows=1, figsize=(10, 6))
    for rr in range(r):
        A = abundances[rr].reshape(H, W)
        # NOTE: Not sure if vmin/vmax should be set...
        ax[rr].imshow(A, vmin=0.0, vmax=1.0, cmap="viridis")

    plt.tight_layout()
    plt.savefig("abundances.png")
