"""Noise generation logic."""

from logging import getLogger
from dataclasses import dataclass

import numpy as np

logs = getLogger(__name__)


@dataclass
class AdditiveWhiteGaussianNoise:  # AWGN
    SNR: float

    def noisify(self, Y):
        """Compute sigmas for the desired SNR given a flattened input HSI Y."""
        logs.debug(f"Y shape => {Y.shape}")
        assert len(Y.shape) == 2
        p, n = Y.shape
        logs.info(f"Desired SNR => {self.SNR}")

        ###########
        # Fitting #
        ###########
        if self.SNR == 0.0:  # No added noise!
            sigmas = np.zeros(p)
            logs.info("No noise to be added...")
        else:
            assert self.SNR > 0.0, "SNR must be strictly positive"
            # Uniform across bands
            sigmas = np.ones(p)
            # Normalization
            sigmas /= np.linalg.norm(sigmas)
            logs.debug(f"Sigmas after normalization: {np.round(sigmas[0], 3)}")
            # Compute sigma mean based on SNR
            num = (Y**2).sum() / n
            denom = 10 ** (self.SNR / 10)
            sigmas_mean = np.sqrt(num / denom)
            logs.debug(f"Sigma mean based on SNR: {np.round(sigmas_mean, 3)}")
            # Noise variance
            sigmas *= sigmas_mean
            logs.debug(f"Final sigmas value: {np.round(sigmas[0], 3)}")

        #############
        # Transform #
        #############
        noise = np.diag(sigmas) @ np.random.randn(p, n)

        # Return additive noise
        return Y + noise
