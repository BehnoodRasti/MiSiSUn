from logging import getLogger
from dataclasses import dataclass
import time

import numpy as np
from skimage.segmentation import slic

from .base import BaseUnmixingModel
from .sunsal import SUnSAL, SUnSAL_SpReg

logs = getLogger(__name__)


@dataclass
class MUA_SLIC(BaseUnmixingModel):
    lambda1: float = 0.0
    lambda2: float = 0.0
    beta: float = 0.0
    slic_size: int = 0
    slic_reg: float = 0.0

    def compute_abundances(
        self,
        Y,
        D,
        H,
        W,
        *args,
        **kwargs,
    ):
        tic = time.time()
        L, N = Y.shape
        LD, M = D.shape
        assert L == LD, "Inconsistent number of channels for D and Y"
        # Reshape into image
        Y_img = Y.T.reshape(H, W, L)

        # Rescale the data
        # NOTE Is the following step necessary?
        scale_factor = np.mean(np.sqrt(np.sum(Y_img**2, 2)))
        Y_img_rescaled = Y_img / scale_factor

        # # Compute superpixels
        labels = slic(
            Y_img_rescaled,
            n_segments=self.slic_size,
            compactness=self.slic_reg,
            channel_axis=2,
            enforce_connectivity=False,
        )
        n_superpixels = labels.max() + 1
        logs.debug(f"{n_superpixels} superpixels used in SLIC")

        # Average all pixels inside each superpixel
        avg_superpixel = np.zeros((n_superpixels, L))
        for ii in range(n_superpixels):
            indices = np.argwhere(labels == ii)
            rowi = indices[:, 0]
            coli = indices[:, 1]
            for jj in range(len(rowi)):
                if jj == 0:
                    avg_superpixel[ii] = (1 / len(rowi)) * Y_img[rowi[jj], coli[jj], :]
                else:
                    avg_superpixel[ii] = (
                        avg_superpixel[ii, :]
                        + (1 / len(rowi)) * Y_img[rowi[jj], coli[jj], :]
                    )

        logs.debug(f" AVG superpixel shape => {avg_superpixel.shape}")

        # Unmix each superpixel individually
        sunsal = SUnSAL(
            AL_iters=2000,
            lambd=self.lambda1,
            positivity=True,
            addone=False,
            tol=1e-4,
        )
        X = sunsal.compute_abundances(np.squeeze(avg_superpixel).T, D)
        logs.info(f"X shape => {X.shape}")

        # Reattribute abundances for the entire matrix
        A0 = np.zeros((H, W, M))
        for label in range(n_superpixels):
            A0[labels == label] = X[:, label]

        # Unmix using spatially regularized SUnSAL
        sunsal_spreg = SUnSAL_SpReg(
            AL_iters=2000,
            lambd=self.lambda2,
            positivity=True,
            addone=False,
            tol=1e-4,
            beta=self.beta,
        )
        X_hat = A0.transpose(2, 0, 1).reshape(M, -1)
        Ahat = sunsal_spreg.compute_abundances(Y, D, X_hat=X_hat)
        tac = time.time()
        self.register_time(tac - tic)
        logs.info(self.processing_time)
        return Ahat
