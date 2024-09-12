"""Unmixing model logic implementation."""

from dataclasses import dataclass
from logging import getLogger

import numpy as np
import numpy.linalg as LA
# from sklearn.cluster import KMeans

logs = getLogger(__name__)


@dataclass
class BatchVCA:
    Y: np.ndarray
    r: int
    bundles_nbr: int = 10
    percentage: int = 10

    def apply(self):
        p, n = self.Y.shape
        rng = np.random.default_rng()
        nb_spectra = np.floor(n * self.percentage / 100)
        # Select sub-samples from HSImage
        # NOTE: In that scenario some spectra may be picked several times! This is different from L. Drumetz implementation that discards the pixels used.
        bundles = [
            self.Y[:, rng.choice(n, size=nb_spectra, replace=False)]
            for _ in range(self.bundles_nbr)
        ]
        # Use VCA on sub-samples
        endmembers = []
        for bundle in bundles:
            pixels, indices = VCA().extract_endmembers(Y=bundle, r=self.r)
            endmembers.append(pixels)
        # Convert to np array
        E = np.array(endmembers)
        # Clustering on extracted endmembers
        # kmeans = KMeans(n_clusters=self.r, random_state=0, n_init="auto").fit(E.T)
        # TODO: Use cosine distance!
        return E


@dataclass
class VCA:
    seed: int = 0
    snr_input: float = 0.0

    def extract_endmembers(
        self,
        Y,
        r,
        *args,
        **kwargs,
    ):
        """
        Vertex Component Analysis

        This code is a translation of a matlab code provided by
        Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
        available at http://www.lx.it.pt/~bioucas/code.htm
        under a non-specified Copyright (c)
        Translation of last version at 22-February-2018
        (Matlab version 2.1 (7-May-2004))

        more details on:
        Jose M. P. Nascimento and Jose M. B. Dias
        "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
        submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
        """
        p, n = Y.shape
        generator = np.random.default_rng(seed=self.seed)

        #############################################
        # SNR Estimates
        #############################################

        if self.snr_input == 0:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m  # data with zero-mean
            Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(n))[0][
                :, :r
            ]  # computes the R-projection matrix
            x_p = np.dot(Ud.T, Y_o)  # project the zero-mean data onto p-subspace

            SNR = self.estimate_snr(Y, y_m, x_p)

            logs.info(f"SNR estimated = {SNR}[dB]")
        else:
            SNR = self.snr_input
            logs.info(f"input SNR = {SNR}[dB]\n")

        SNR_th = 15 + 10 * np.log10(r)
        #############################################
        # Choosing Projective Projection or
        #          projection to p-1 subspace
        #############################################

        if SNR < SNR_th:
            logs.info("... Select proj. to R-1")

            d = r - 1
            if self.snr_input == 0:  # it means that the projection is already computed
                Ud = Ud[:, :d]
            else:
                y_m = np.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m  # data with zero-mean

                Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(n))[0][
                    :, :d
                ]  # computes the p-projection matrix
                x_p = np.dot(Ud.T, Y_o)  # project thezeros mean data onto p-subspace

            Yp = np.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

            x = x_p[:d, :]  #  x_p =  Ud.T * Y_o is on a R-dim subspace
            c = np.amax(np.sum(x**2, axis=0)) ** 0.5
            y = np.vstack((x, c * np.ones((1, n))))
        else:
            logs.info("... Select the projective proj.")

            d = r
            Ud = LA.svd(np.dot(Y, Y.T) / float(n))[0][
                :, :d
            ]  # computes the p-projection matrix

            x_p = np.dot(Ud.T, Y)
            Yp = np.dot(
                Ud, x_p[:d, :]
            )  # again in dimension L (note that x_p has no null mean)

            x = np.dot(Ud.T, Y)
            u = np.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
            y = x / np.dot(u.T, x)

        #############################################
        # VCA algorithm
        #############################################

        indices = np.zeros((r), dtype=int)
        A = np.zeros((r, r))
        A[-1, 0] = 1

        for i in range(r):
            w = generator.random(size=(r, 1))
            f = w - np.dot(A, np.dot(LA.pinv(A), w))
            f = f / np.linalg.norm(f)

            v = np.dot(f.T, y)

            indices[i] = np.argmax(np.absolute(v))
            A[:, i] = y[:, indices[i]]  # same as x(:,indice(i))

        E = Yp[:, indices]

        logs.debug(f"Indices chosen to be the most pure: {indices}")

        return E, indices

    @staticmethod
    def estimate_snr(Y, r_m, x):
        p, n = Y.shape  # L number of bands (channels), N number of pixels
        r, n = x.shape  # p number of endmembers (reduced dimension)

        P_y = np.sum(Y**2) / float(n)
        P_x = np.sum(x**2) / float(n) + np.sum(r_m**2)
        snr_est = 10 * np.log10((P_x - r / p * P_y) / (P_y - P_x))

        return snr_est
