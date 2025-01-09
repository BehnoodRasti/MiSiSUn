import logging
import argparse
from pathlib import Path

import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

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
    # indices = [296, 419, 464]  # Specific indices to plot

    # fig, ax = plt.subplots(ncols=len(indices), nrows=1, figsize=(10, 6))
    # for i, rr in enumerate(indices):
    #     A = abundances[rr].reshape(H, W)
    #     # NOTE: Not sure if vmin/vmax should be set...
    #     ax[i].imshow(A, vmin=0.0, vmax=1.0, cmap="viridis")

    # plt.tight_layout()
    # plt.savefig("outputs/abundances.png")
    r = len(abundances)
    ncols = 6
    nrows = math.ceil(r / ncols)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    ax = ax.flatten()  # Flatten the array of axes for easy iteration

    for rr in range(r):
        A = abundances[rr].reshape(H, W)
        im = ax[rr].imshow(A, cmap="viridis")
        divider = make_axes_locatable(ax[rr])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=16)  # Set the font size for the colorbar
        im.set_clim(vmin=A.min(), vmax=A.max())
        ax[rr].axis("off")

    # Hide any unused subplots
    for i in range(r, len(ax)):
        fig.delaxes(ax[i])

    plt.tight_layout()
    plt.savefig("abundances.png")
    # fig, ax = plt.subplots(ncols=r, nrows=1, figsize=(10, 6))
    # for rr in range(r):
    #     A = abundances[rr].reshape(H, W)
    #     im = ax[rr].imshow(A, cmap="viridis")
    #     divider = make_axes_locatable(ax[rr])
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     fig.colorbar(im, cax=cax)
    #     cbar = fig.colorbar(im, cax=cax)
    #     cbar.ax.tick_params(labelsize=4) 
    #     im.set_clim(vmin=A.min(), vmax=A.max())
    #     ax[rr].axis("off")
    #     # A = abundances[rr].reshape(H, W)
    #     # im = ax[rr].imshow(A, vmin=0.0, vmax=1.0, cmap="viridis")
    #     # divider = make_axes_locatable(ax[rr])
    #     # cax = divider.append_axes("right", size="5%", pad=0.05)
    #     # fig.colorbar(im, cax=cax)
    #     # ax[rr].axis("off")
    #     #
    #     # A = abundances[rr].reshape(H, W)
    #     # # NOTE: Not sure if vmin/vmax should be set...
    #     # im = ax[rr].imshow(A, vmin=0.0, vmax=1.0, cmap="viridis")
    #     # fig.colorbar(im, ax=ax[rr])
    #     # # ax[rr].imshow(A, vmin=0.0, vmax=1.0, cmap="viridis")
    #     # ax[rr].axis("off")
    # plt.tight_layout()
    # plt.savefig("abundances.png")
