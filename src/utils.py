"""Utility routines."""

from logging import getLogger
import numpy as np

logs = getLogger(__name__)


def SVD_projection(Y, r):
    logs.debug(f"Y shape => {Y.shape}")
    V, SS, U = np.linalg.svd(Y, full_matrices=False)
    PC = np.diag(SS) @ U
    denoised_image_reshape = V[:, :r] @ PC[:r]
    logs.debug(f"Projected Y shape => {denoised_image_reshape}")
    return np.clip(denoised_image_reshape, 0.0, 1.0)
