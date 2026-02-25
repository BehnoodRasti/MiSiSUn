import logging
import argparse
from pathlib import Path

import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import numpy as np

# CUPRITE DIMENSIONS
H = 250
W = 191

def parse_indices(text, total):
    if not text:
        return None
    idxs = [int(s) for s in text.split(",") if s.strip() != ""]
    return [i for i in idxs if i < total]


def plot_abundances(abundances, selection):
    r = len(selection)
    ncols = 6
    nrows = math.ceil(r / ncols)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    ax = ax.flatten()  # Flatten the array of axes for easy iteration

    for plot_idx, rr in enumerate(selection):
        A = abundances[rr].reshape(H, W)
        im = ax[plot_idx].imshow(A, cmap="viridis")
        divider = make_axes_locatable(ax[plot_idx])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=16)  # Set the font size for the colorbar
        im.set_clim(vmin=A.min(), vmax=A.max())
        ax[plot_idx].axis("off")

    # Hide any unused subplots
    for i in range(r, len(ax)):
        fig.delaxes(ax[i])

    plt.tight_layout()
    plt.savefig("abundances.png")
    logging.info("Saved abundances.png")


def plot_endmembers(db_hat, selection):
    db = np.array(db_hat)
    # Expect shape (bands, k). If we got (k, bands), fix it.
    if db.shape[0] < db.shape[1]:
        db = db.T
    bands, k = db.shape

    fig, axes = plt.subplots(
        nrows=math.ceil(len(selection) / 3), ncols=3, figsize=(12, 4 * math.ceil(len(selection) / 3))
    )
    axes = np.atleast_1d(axes).flatten()
    for plot_idx, rr in enumerate(selection):
        ax = axes[plot_idx]
        ax.plot(db[:, rr])
        ax.set_title(f"Endmember {rr}")
        ax.set_xlabel("Band")
        ax.set_ylabel("Reflectance (a.u.)")
        ax.grid(True, alpha=0.3)
    # Hide unused subplots
    for i in range(len(selection), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig("endmembers.png")
    logging.info("Saved endmembers.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", "-p", type=str, help="Path to results.mat", required=True)
    parser.add_argument(
        "--mode",
        choices=["abundances", "endmembers", "both"],
        default="abundances",
        help="What to plot",
    )
    parser.add_argument(
        "--indices",
        type=str,
        help="Comma-separated abundance indices to plot (0-based). Example: 0,1,2",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Plot only the first K abundances (if --indices not set)",
    )
    parser.add_argument(
        "--endmember-indices",
        type=str,
        help="Comma-separated endmember indices to plot (0-based). Example: 0,1,2",
    )
    parser.add_argument(
        "--endmember-topk",
        type=int,
        default=None,
        help="Plot only the first K endmembers (if --endmember-indices not set)",
    )

    args = parser.parse_args()

    result_path = Path(args.path)
    assert result_path.is_file()

    result = sio.loadmat(result_path)

    if args.mode in ("abundances", "both"):
        if "A_hat" not in result:
            raise KeyError("A_hat not found in results file.")
        abundances = result["A_hat"]
        r_total = len(abundances)
        selection = parse_indices(args.indices, r_total)
        if selection is None:
            if args.topk is not None:
                selection = list(range(min(args.topk, r_total)))
            else:
                selection = list(range(r_total))
        plot_abundances(abundances, selection)

    if args.mode in ("endmembers", "both"):
        db_hat = result.get("DB_hat")
        if db_hat is None:
            raise KeyError("DB_hat not found in results file. Run unmixing with save_DB=true.")
        # Determine component count assuming (bands, k) or (k, bands)
        if db_hat.shape[0] < db_hat.shape[1]:
            k_total = db_hat.shape[0]
        else:
            k_total = db_hat.shape[1]
        em_selection = parse_indices(args.endmember_indices, k_total)
        if em_selection is None:
            if args.endmember_topk is not None:
                em_selection = list(range(min(args.endmember_topk, k_total)))
            else:
                em_selection = list(range(k_total))
        plot_endmembers(db_hat, em_selection)
