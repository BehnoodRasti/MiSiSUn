"""SUnSAL related unmixing models."""

from logging import getLogger
import time
from dataclasses import dataclass

import numpy as np
import numpy.linalg as LA
from scipy.signal import convolve2d

from .base import BaseUnmixingModel

logs = getLogger(__name__)


@dataclass
class SUnSAL(BaseUnmixingModel):
    AL_iters: int = 1000  # iterations
    lambd: float = 0.0  # regularization parameter
    verbose: bool = True
    positivity: bool = False
    addone: bool = False
    tol: float = 1e-4  # used as stopping criterion
    x0: float = 0.0  # initial point

    def compute_abundances(self, Y, D, *args, **kwargs):
        tic = time.time()
        LD, M = D.shape
        L, N = Y.shape

        assert L == LD, "Inconsistent number of channels for D and Y"

        # Lambda for all pixels
        lambd = self.lambd * np.ones((M, N))

        # Compute mean norm
        # NOTE: This typo led to better results
        # norm_d = np.sqrt(np.mean(D**2)) * (25 * M) / M
        # norm_d = np.sqrt(np.mean(D**2))
        # NOTE: Align with original MATLAB code
        norm_d = np.sqrt(np.mean(D**2))
        logs.debug(f"norm D => {norm_d:.3e}")
        # # Rescale D, Y and lambda
        D = D / norm_d
        Y = Y / norm_d
        lambd = lambd / norm_d**2
        logs.debug(f"lambda value: {np.mean(lambd):.3e}")

        # Least squares
        if np.sum(lambd == 0) and not self.addone and not self.positivity:
            logs.debug("Least Squares")
            Ahat = LA.pinv(D) @ Y
            return Ahat

        # Constrained Least Squares (sum(x) = 1)
        SMALL = 1e-12
        B = np.ones((1, M))
        a = np.ones((1, N))

        if np.sum(lambd == 0) and self.addone and not self.positivity:
            logs.debug("Constrained Least Squares (sum(x) = 1)")
            F = D.T @ D
            # Test if F is invertible
            if LA.cond(F) > SMALL:
                # Compute the solution explicitely
                IF = LA.inv(F)
                Ahat = IF @ D.T @ Y - IF @ B.T @ (LA.inv(B @ IF @ B.T)) @ (
                    B @ IF @ D.T @ Y - a
                )
                return Ahat

        # Constants and initialization
        mu_AL = 0.01
        mu = 10 * np.mean(lambd) + mu_AL
        # mu = mu_AL
        logs.debug(f"mu initial value: {mu:.3e}")

        UF, sF, VF = LA.svd(D.T @ D)
        # SF = np.diag(sF)
        IF = UF @ (np.diag(1 / (sF + mu))) @ UF.T

        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
        x_aux = Aux @ a
        IF1 = IF - Aux @ B @ IF
        yy = D.T @ Y

        # Initializations
        if self.x0 == 0:
            x = IF @ D.T @ Y
        else:
            x = self.x0

        z = x
        # Scaled Lagrange Multipliers
        d = 0 * z

        # AL iterations
        tol1 = np.sqrt(N * M) * self.tol
        tol2 = np.sqrt(N * M) * self.tol
        logs.debug(f"tolerance => {tol1:.3e}")
        i = 1
        res_p = float("inf")
        res_d = float("inf")
        mu_changed = 0

        # Constrained Least Squares (CLS) X >= 0

        if np.sum(lambd == 0) and not self.addone:
            logs.debug("Constrained Least Squares (x >= 0)")
            while (i <= self.AL_iters) and (
                (np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)
            ):
                # Save z to be used later
                if i % 10 == 1:
                    z0 = z

                # Minimize w.r.t. z
                z = np.maximum(x - d, 0)
                # Minimize w.r.t. x
                x = IF @ (yy + mu * (z + d))
                # Lagrange multipliers update
                d = d - (x - z)

                # Update mu to keep primal and dual residuals within a factor of 10
                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(x - z)
                    # dual residual
                    res_d = mu * LA.norm(z - z0)
                    if self.verbose:
                        logs.info(f"i = {i}, res_p = {res_p:.3e}, res_d = {res_d:.3e}")

                    # update mu
                    if res_p > 10 * res_d:
                        mu = mu * 2
                        d = d / 2
                        mu_changed = 1

                    elif res_d > 10 * res_p:
                        mu = mu / 2
                        d = d * 2
                        mu_changed = 1

                    if mu_changed:
                        logs.debug(f"mu changed ({i}) => {mu}")
                        # Update IF and IF1
                        IF = UF @ np.diag(1 / (sF + mu)) @ UF.T
                        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
                        x_aux = Aux @ a
                        IF1 = IF - Aux @ B @ IF
                        mu_changed = 0

                i += 1

        # Fully Constraint Least Squares
        elif np.sum(lambd == 0) and self.addone:
            logs.debug("Fully Constrained Least Squares")
            while (i <= self.AL_iters) and (
                (np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)
            ):
                # Save z to be used later
                if i % 10 == 1:
                    z0 = z

                # Minimize w.r.t. z
                z = np.maximum(x - d, 0)
                # Minimize w.r.t. x
                x = IF1 @ (yy + mu * (z + d)) + x_aux
                # Lagrange multipliers update
                d = d - (x - z)

                # Update mu to keep primal and dual residuals within a factor of 10
                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(x - z)
                    # dual residual
                    res_d = mu * LA.norm(z - z0)
                    if self.verbose:
                        logs.info(f"i = {i}, res_p = {res_p:.3e}, res_d = {res_d:.3e}")

                    # update mu
                    if res_p > 10 * res_d:
                        mu = mu * 2
                        d = d / 2
                        mu_changed = 1

                    elif res_d > 10 * res_p:
                        mu = mu / 2
                        d = d * 2
                        mu_changed = 1

                    if mu_changed:
                        # Update IF and IF1
                        IF = UF @ np.diag(1 / (sF + mu)) @ UF.T
                        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
                        x_aux = Aux @ a
                        IF1 = IF - Aux @ B @ IF
                        mu_changed = 0

                i += 1

        # Generic SUnSAL
        else:
            logs.debug("Generic SUnSAL")

            def softthresh(x, th):
                return np.sign(x) * np.maximum(np.abs(x) - th, 0)

            while (i <= self.AL_iters) and (
                (np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)
            ):
                if i % 10 == 1:
                    z0 = z

                # Minimize w.r.t. z
                z = softthresh(x - d, lambd / mu)
                # Test for positivity
                if self.positivity:
                    z = np.maximum(z, 0)

                # Test of Sum-to-one
                if self.addone:
                    x = IF1 @ (yy + mu * (z + d)) + x_aux
                else:
                    x = IF @ (yy + mu * (z + d))

                # Lagrange multipliers update
                d = d - (x - z)

                # Update mu to keep primal and dual residuals within a factor of 10
                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(x - z)
                    # dual residual
                    res_d = mu * LA.norm(z - z0)
                    if self.verbose:
                        logs.info(f"i = {i}, res_p = {res_p:.3e}, res_d = {res_d:.3e}")

                    # update mu
                    if res_p > 10 * res_d:
                        mu = mu * 2
                        d = d / 2
                        mu_changed = 1

                    elif res_d > 10 * res_p:
                        mu = mu / 2
                        d = d * 2
                        mu_changed = 1

                    if mu_changed:
                        # Update IF and IF1
                        logs.debug(f"mu changed ({i}) => {mu}")
                        IF = UF @ np.diag(1 / (sF + mu)) @ UF.T
                        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
                        x_aux = Aux @ a
                        IF1 = IF - Aux @ B @ IF
                        mu_changed = 0

                i += 1

        Ahat = z
        tac = time.time()
        self.register_time(tac - tic)
        logs.info(self.processing_time)
        return Ahat


@dataclass
class SUnSAL_SpReg(BaseUnmixingModel):
    AL_iters: int = 1000  # iterations
    lambd: float = 0.0  # regularization parameter
    verbose: bool = True
    positivity: bool = False
    addone: bool = False
    tol: float = 1e-4  # used as stopping criterion
    x0: float = 0.0  # initial point
    beta: float = 0.0  # spatial regularizer

    def compute_abundances(self, Y, D, X_hat, *args, **kwargs):
        tic = time.time()
        LD, M = D.shape
        L, N = Y.shape
        MX, NX = X_hat.shape

        assert LD == L, "Inconsistent number of channels for D and Y"
        assert MX == M, "Inconsistent number of atoms for D and X"
        assert N == NX, "Inconsistent number of pixels for X and Y"

        # Lambda for all pixels
        lambd = self.lambd * np.ones((M, N))

        # Compute mean norm
        norm_y = np.sqrt(np.mean(Y**2))
        logs.debug(f"Norm Y => {norm_y:.3e}")
        # Rescale D, Y and lambda
        Y = Y / norm_y
        D = D / norm_y
        lambd = lambd / (norm_y**2)

        # Least squares
        if (
            np.sum(lambd == 0)
            and not self.addone
            and not self.positivity
            and self.beta == 0
        ):
            logs.debug("Least Squares")
            Ahat = LA.pinv(D) @ Y
            return Ahat

        # Constrained Least Squares (sum(x) = 1)
        SMALL = 1e-12
        B = np.ones((1, M))
        a = np.ones((1, N))

        if (
            np.sum(lambd == 0)
            and self.addone
            and not self.positivity
            and self.beta == 0
        ):
            logs.debug("Constrained Least Squares (sum(x) = 1)")
            F = D.T @ D
            # Test if F is invertible
            if LA.cond(F) > SMALL:
                # Compute the solution explicitely
                IF = LA.inv(F)
                Ahat = IF @ D.T @ Y - IF @ B.T @ (LA.inv(B @ IF @ B.T)) @ (
                    B @ IF @ D.T @ Y - a
                )
                return Ahat

        # Constants and initialization
        mu_AL = 0.01
        mu = 10 * np.mean(lambd) + mu_AL

        logs.debug(f"mu initial value: {mu:.3e}")

        UF, sF, VF = LA.svd(D.T @ D)
        # SF = np.diag(sF)
        IF = UF @ (np.diag(1 / (sF + mu + self.beta))) @ UF.T

        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
        x_aux = Aux @ a
        IF1 = IF - Aux @ B @ IF
        yy = D.T @ Y

        # Initializations
        if self.x0 == 0:
            x = IF @ D.T @ Y
        else:
            x = self.x0

        z = x
        # Scaled Lagrange Multipliers
        d = 0 * z

        # AL iterations
        tol1 = np.sqrt(N * M) * self.tol
        tol2 = np.sqrt(N * M) * self.tol
        logs.debug(f"tolerance => {tol1:.3e}")
        i = 1
        res_p = float("inf")
        res_d = float("inf")
        mu_changed = 0

        # Constrained Least Squares (CLS) X >= 0

        if np.sum(lambd == 0) and not self.addone:
            logs.debug("Constrained Least Squares (x >= 0) + spatial reg.")
            while (i <= self.AL_iters) and (
                (np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)
            ):
                # Save z to be used later
                if i % 10 == 1:
                    z0 = z

                # Minimize w.r.t. z
                z = np.maximum(x - d, 0)
                # Minimize w.r.t. x
                x = IF @ (yy + mu * (z + d) + self.beta * X_hat)
                # Lagrange multipliers update
                d = d - (x - z)

                # Update mu to keep primal and dual residuals within a factor of 10
                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(x - z)
                    # dual residual
                    res_d = mu * LA.norm(z - z0)
                    if self.verbose:
                        logs.info(f"i = {i}, res_p = {res_p:.3e}, res_d = {res_d:.3e}")

                    # update mu
                    if res_p > 10 * res_d:
                        mu = mu * 2
                        d = d / 2
                        mu_changed = 1

                    elif res_d > 10 * res_p:
                        mu = mu / 2
                        d = d * 2
                        mu_changed = 1

                    if mu_changed:
                        logs.debug(f"mu changed ({i}) => {mu}")
                        # Update IF and IF1
                        IF = UF @ np.diag(1 / (sF + mu + self.beta)) @ UF.T
                        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
                        x_aux = Aux @ a
                        IF1 = IF - Aux @ B @ IF
                        mu_changed = 0

                i += 1

        # Fully Constraint Least Squares
        elif np.sum(lambd == 0) and self.addone:
            logs.debug("Fully Constrained Least Squares + spatial reg.")
            while (i <= self.AL_iters) and (
                (np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)
            ):
                # Save z to be used later
                if i % 10 == 1:
                    z0 = z

                # Minimize w.r.t. z
                z = np.maximum(x - d, 0)
                # Minimize w.r.t. x
                x = IF1 @ (yy + mu * (z + d)) + x_aux
                # Lagrange multipliers update
                d = d - (x - z)

                # Update mu to keep primal and dual residuals within a factor of 10
                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(x - z)
                    # dual residual
                    res_d = mu * LA.norm(z - z0)
                    if self.verbose:
                        logs.info(f"i = {i}, res_p = {res_p:.3e}, res_d = {res_d:.3e}")

                    # update mu
                    if res_p > 10 * res_d:
                        mu = mu * 2
                        d = d / 2
                        mu_changed = 1

                    elif res_d > 10 * res_p:
                        mu = mu / 2
                        d = d * 2
                        mu_changed = 1

                    if mu_changed:
                        # Update IF and IF1
                        IF = UF @ np.diag(1 / (sF + mu)) @ UF.T
                        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
                        x_aux = Aux @ a
                        IF1 = IF - Aux @ B @ IF
                        mu_changed = 0

                i += 1

        # Generic SUnSAL
        else:
            logs.debug("Generic SUnSAL + spatial reg.")

            def softthresh(x, th):
                return np.sign(x) * np.maximum(np.abs(x) - th, 0)

            while (i <= self.AL_iters) and (
                (np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)
            ):
                if i % 10 == 1:
                    z0 = z

                # Minimize w.r.t. z
                z = softthresh(x - d, lambd / mu)
                # Test for positivity
                if self.positivity:
                    z = np.maximum(z, 0)

                # Test of Sum-to-one
                if self.addone:
                    x = IF1 @ (yy + mu * (z + d)) + x_aux
                else:
                    x = IF @ (yy + mu * (z + d) + self.beta * X_hat)

                # Lagrange multipliers update
                d = d - (x - z)

                # Update mu to keep primal and dual residuals within a factor of 10
                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(x - z)
                    # dual residual
                    res_d = mu * LA.norm(z - z0)
                    if self.verbose:
                        logs.info(f"i = {i}, res_p = {res_p:.3e}, res_d = {res_d:.3e}")

                    # update mu
                    if res_p > 10 * res_d:
                        mu = mu * 2
                        d = d / 2
                        mu_changed = 1

                    elif res_d > 10 * res_p:
                        mu = mu / 2
                        d = d * 2
                        mu_changed = 1

                    if mu_changed:
                        # Update IF and IF1
                        logs.debug(f"mu changed ({i}) => {mu}")
                        IF = UF @ np.diag(1 / (sF + mu + self.beta)) @ UF.T
                        Aux = IF @ B.T @ (LA.inv(B @ IF @ B.T))
                        x_aux = Aux @ a
                        IF1 = IF - Aux @ B @ IF
                        mu_changed = 0

                i += 1

        Ahat = z
        tac = time.time()
        self.register_time(tac - tic)
        logs.info(self.processing_time)
        return Ahat


@dataclass
class S2WSU(BaseUnmixingModel):
    AL_iters: int = 5
    lambd: float = 0.0
    verbose: bool = True
    tol: float = 1e-4
    x0: float = 0.0

    @staticmethod
    def soft(b, t):
        max_b = np.maximum(np.abs(b) - t, 0)
        return b * (max_b / (max_b + t))

    def compute_abundances(self, Y, D, H, W, *args, **kwargs):
        tic = time.time()
        LD, M = D.shape
        L, N = Y.shape

        assert L == LD, "Inconsistent number of channels for D and Y"

        lambd = self.lambd

        # # Compute mean norm
        # NOTE Legacy code
        # norm_d = np.sqrt(np.mean(D**2))
        # logger.debug(f"Norm D => {norm_d:.3e}")
        # # Rescale D, Y and lambda
        # D = D / norm_d
        # Y = Y / norm_d
        # lambd = lambd / norm_d**2
        logs.debug(f"Lambda initial value => {lambd:.3e}")

        # Constants and initialization
        mu = 0.5
        logs.debug(f"Mu initial value => {mu:.3e}")

        UF, sF, VF = LA.svd(D.T @ D)
        IF = UF @ (np.diag(1 / (sF + mu))) @ UF.T

        AA = LA.inv(D.T @ D + 2 * np.eye(M))

        # Initializations
        if self.x0 == 0:
            x = IF @ D.T @ Y
        else:
            x = self.x0

        u = x
        v1 = D @ x
        v2 = x
        v3 = x
        # # Scaled Lagrange Multipliers
        d1 = 0 * v1
        d2 = 0 * v2
        d3 = 0 * v3

        # # AL iterations
        tol = np.sqrt(N * M) * self.tol
        logs.debug(f"Tolerance => {tol:.3e}")
        k = 1
        i = 1
        res_p = float("inf")
        # res_d = float("inf")
        AL_iters2 = 60

        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        kernel[0, 0] = 1 / np.sqrt(2)
        kernel[2, 0] = 1 / np.sqrt(2)
        kernel[0, 2] = 1 / np.sqrt(2)
        kernel[2, 2] = 1 / np.sqrt(2)
        kernel = kernel / (4 + 4 / np.sqrt(2))

        while k <= AL_iters2:
            NU = np.zeros((M, N))
            X2 = np.reshape(v3 - d3, (M, H, W))
            for ii in range(M):
                NU[ii] = convolve2d(X2[ii], kernel, mode="same").flatten()

            w = 1 / (0.01 + np.abs(NU))

            NU2 = LA.norm(v3 - d3, axis=1, keepdims=True)
            w1 = w / NU2

            while (i <= self.AL_iters) and np.abs(res_p) > tol:
                # Save u to be used later
                # if i % 10 == 1:
                #    u0 = u

                # Minimize w.r.t. u
                u = AA @ (D.T @ (v1 + d1) + v2 + d2 + v3 + d3)

                # Minimize w.r.t. v1
                v1 = (Y + mu * (D @ u - d1)) / (1 + mu)

                # Minimize w.r.t. v2
                v2 = np.maximum(u - d2, 0)

                # Minimize w.r.t. v3
                v3 = self.soft(u - d3, (lambd / mu) * w1)

                # Lagrange multipliers update
                d1 = d1 - D @ u + v1
                d2 = d2 - u + v2
                d3 = d3 - u + v3

                if i % 10 == 1:
                    # primal residual
                    res_p = LA.norm(D @ u - v1) + LA.norm(u - v2) + LA.norm(u - v3)
                    if self.verbose:
                        logs.info(f"k = {k}, i = {i}, res_p = {res_p:.3e}")

                i += 1

            i = 1
            k += 1

        Ahat = u

        tac = time.time()
        self.register_time(tac - tic)
        logs.info(self.processing_time)

        return Ahat


@dataclass
class CLSUnSAL(BaseUnmixingModel):
    AL_iters: int = 1000
    lambd: float = 0.0
    verbose: bool = True
    tol: float = 1e-4
    mu: float = 0.1  # lagrangian multiplier
    x0: float = 0.0

    def compute_abundances(self, Y, D, *args, **kwargs):
        tic = time.time()
        LD, M = D.shape
        L, N = Y.shape

        assert L == LD, "Inconsistent number of channels for D and Y"

        lambd = self.lambd

        # # Compute mean norm
        # NOTE Legacy code
        norm_d = np.sqrt(np.mean(D**2))
        logs.debug(f"Norm D => {norm_d:.3e}")
        # Rescale D, Y and lambda
        D = D / norm_d
        Y = Y / norm_d
        lambd = lambd / norm_d**2
        logs.debug(f"Lambda initial value => {lambd:.3e}")

        # Constants and initialization
        # mu = 0.1
        mu = self.mu
        logs.debug(f"Mu initial value => {mu:.3e}")

        UF, sF, VF = LA.svd(D.T @ D)
        IF = UF @ (np.diag(1 / (sF + mu))) @ UF.T

        AA = LA.inv(D.T @ D + 2 * np.eye(M))

        # Initializations
        if self.x0 == 0:
            x = IF @ D.T @ Y
        else:
            x = self.x0

        u = x
        v1 = D @ x
        v2 = x
        v3 = x
        # # Scaled Lagrange Multipliers
        d1 = v1
        d2 = v2
        d3 = v3

        # # AL iterations
        tol = np.sqrt(N * M) * self.tol
        logs.debug(f"Tolerance => {tol:.3e}")
        k = 1
        res_p = float("inf")
        res_d = float("inf")

        while (k <= self.AL_iters) and ((np.abs(res_p) > tol) or (np.abs(res_d) > tol)):
            # Save u to be used later
            if k % 10 == 1:
                u0 = u

            # breakpoint()
            # Minimize w.r.t. u
            # NOTE Legacy (might be faster than solving linear system)
            u = AA @ (D.T @ (v1 + d1) + v2 + d2 + v3 + d3)
            # u = LA.solve(DD, D.T @ (v1 + d1) + v2 + d2 + v3 + d3)

            # Minimize w.r.t. v1
            v1 = (Y + mu * (D @ u - d1)) / (1 + mu)

            # Minimize w.r.t. v2
            def current_fn(b):
                return self.vect_soft_thresh(b, lambd / mu)

            v2 = np.apply_along_axis(current_fn, axis=1, arr=u - d2)

            # Minimize w.r.t. v3
            v3 = np.maximum(u - d3, 0)

            # Lagrange multipliers update
            d1 = d1 - D @ u + v1
            d2 = d2 - u + v2
            d3 = d3 - u + v3

            # Update mu to keep primal and dual residuals within a factor of 10
            if k % 10 == 1:
                # primal residual
                res_p = LA.norm(D @ u - v1) + LA.norm(u - v2) + LA.norm(u - v3)
                # dual residual
                res_d = mu * LA.norm(u - u0)
                if self.verbose:
                    logs.info(
                        f"k = {k}, res_p = {res_p:.3e}, res_d = {res_d:.3e}, mu = {mu:.3e}"
                    )

                # Update mu
                if res_p > 10 * res_d:
                    mu = mu * 2

                if res_d > 10 * res_p:
                    mu = mu / 2

            k += 1

        Ahat = v3
        tac = time.time()
        self.register_time(tac - tic)
        logs.info(self.processing_time)
        return Ahat

    @staticmethod
    def vect_soft_thresh(b, t):
        max_b = np.maximum(LA.norm(b) - t, 0)
        ret = b * (max_b) / (max_b + t)
        return ret
