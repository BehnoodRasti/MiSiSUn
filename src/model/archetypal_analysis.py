"""Archetypal analysis models."""

from dataclasses import dataclass
from logging import getLogger
import time
from math import sqrt

import numpy as np
import spams
from tqdm import tqdm
import torch


from .base import BaseUnmixingModel

logs = getLogger(__name__)
EPS = 1e-10


@dataclass
class SUnAA(BaseUnmixingModel):
    T: int = 500
    low_rank: bool = False

    def compute_abundances(self, Y, D, r, *args, **kwargs):
        self.r = r

        def loss(a, b):
            return 0.5 * ((Y - (D @ b) @ a) ** 2).sum()

        def update_B(a, b):
            R = Y - (D @ b) @ a
            for jj in range(self.r):
                z_j = D @ b[:, jj]
                norm_aj = np.linalg.norm(a[jj])
                if norm_aj < 1e-10:
                    ZZ = z_j
                else:
                    ZZ = (R @ a[jj]) / (norm_aj**2) + z_j
                bb = spams.decompSimplex(np.asfortranarray(ZZ[:, np.newaxis]), DD)
                b[:, jj] = np.squeeze(bb.todense())
                R = R + (z_j - D @ b[:, jj])[:, np.newaxis] @ a[jj][np.newaxis, :]
            return b

        tic = time.time()

        _, N = Y.shape
        _, N_atoms = D.shape

        YY = np.asfortranarray(Y)
        DD = np.asfortranarray(D)
        B = (1 / N_atoms) * np.ones((N_atoms, self.r))
        A = (1 / self.r) * np.ones((self.r, N))

        logs.info(f"Initial loss => {loss(A, B):.2f}")

        progress = tqdm(range(self.T))
        for pp in progress:
            # B = update_B(A, B)
            B = update_B(A, B)
            # logger.debug(f"B update => {loss(A, B):.2f}")
            A = np.array(spams.decompSimplex(YY, np.asfortranarray(D @ B)).todense())
            # logger.debug(f"A update => {loss(A, B):.2f}")
            progress.set_postfix_str(f"loss={loss(A, B):.2f}")
            # if np.isnan(loss(A, B)):
            #    # Restart
            #    pp = 0
            #    B = (1 / N_atoms) * np.ones((N_atoms, self.p))
            #    A = (1 / self.p) * np.ones((self.p, N))

        tac = time.time()
        self.register_time(tac - tic)
        logs.info(self.processing_time)

        logs.info(f"Final loss => {loss(A, B):.2f}")

        self.E_hat = D @ B
        self.B = B
        self.A_lowrank = A  # low-rank abundances (i.e. $p$ endmembers)

        if self.low_rank:
            return A
        return B @ A  # redundant/full abundances


@dataclass
class FaSUn(BaseUnmixingModel):
    mu1: float = 100.0
    mu2: float = 10.0
    mu3: float = 1.0
    TA: int = 5
    TB: int = 5
    T: int = 5000
    low_rank: bool = False

    @torch.no_grad()  # NOTE: No gradients needed
    def compute_abundances(
        self,
        Y,
        D,
        r,
        *args,
        **kwargs,
    ):
        # GPU access
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        # Problem dimensions
        p, n = Y.shape
        m = D.shape[1]

        def loss(a, b):
            return 0.5 * ((Y - (D @ b) @ a) ** 2).sum()

        # Timing
        tic = time.time()

        A = (1 / r) * torch.ones((r, n))
        B = (1 / m) * torch.ones((m, r))
        L1 = torch.zeros((r, n))
        L2 = torch.zeros((m, r))
        L3 = torch.zeros((p, r))
        S1 = L1
        S2 = L2
        S3 = L3
        Y = torch.Tensor(Y)
        D = torch.Tensor(D)
        # Send matrices on GPU
        D = D.to(device)
        Y = Y.to(device)
        A = A.to(device)
        B = B.to(device)
        S1 = S1.to(device)
        S2 = S2.to(device)
        S3 = S3.to(device)
        L1 = L1.to(device)
        L2 = L2.to(device)
        L3 = L3.to(device)
        eye_r = torch.eye(r).to(device)
        eye_m = torch.eye(m).to(device)
        ones_r = torch.ones(r).to(device)
        ones_m = torch.ones(m).to(device)
        ones_n = torch.ones(n).to(device)

        Q1inv = self.mu3 * D.t() @ D + self.mu2 * eye_m
        Z1 = torch.linalg.solve(Q1inv, ones_m)
        c1 = -1 / torch.dot(ones_m, Z1)

        U1 = eye_m + c1 * torch.outer(Z1, ones_m)
        V1 = c1 * torch.outer(Z1, ones_r)

        Initloss = loss(A, B)
        logs.info(f"Initial loss => {Initloss:.3e}")
        progress = tqdm(range(self.T))
        for ii in progress:
            updateloss = loss(A, B)
            progress.set_postfix_str(f"loss={updateloss:.4e}")

            Q2inv = self.mu3 * S3.t() @ S3 + self.mu1 * eye_r
            Z2 = torch.linalg.solve(Q2inv, ones_r)
            c2 = -1 / torch.dot(ones_r, Z2)
            U2 = eye_r + c2 * torch.outer(Z2, ones_r)
            V2 = c2 * torch.outer(Z2, ones_n)

            for jj in range(self.TA):
                WA = S3.t() @ Y + self.mu1 * (S1 - L1)
                A = U2 @ torch.linalg.solve(Q2inv, WA) - V2
                S1 = A + L1
                S1[S1 <= 0] = 0
                L1 = L1 + A - S1

            Q3inv = A @ A.t() + self.mu3 * eye_r

            for jj in range(self.TB):
                WB = self.mu3 * D.t() @ (S3 - L3) + self.mu2 * (S2 - L2)
                B = U1 @ torch.linalg.solve(Q1inv, WB) - V1
                S2 = B + L2
                S2[S2 <= 0] = 0
                S3 = torch.linalg.solve(
                    Q3inv, Y @ A.t() + self.mu3 * (D @ B + L3), left=False
                )
                L2 = L2 + B - S2
                L3 = L3 + D @ B - S3

        tac = time.time()
        self.register_time(tac - tic)
        logs.info(self.processing_time)
        logs.info(f"Final loss => {loss(A, B):.2e}")

        A = A.cpu().numpy()
        B = B.cpu().numpy()
        # return A, B
        if self.low_rank:
            return A
        return B @ A  # full-rank abundances!


@dataclass
class MiSiSUn(BaseUnmixingModel):
    mu1: float = 100.0
    mu2: float = 10.0
    mu3: float = 1.0
    TA: int = 5
    TB: int = 5
    T: int = 5000
    lamb: float = 0.001
    low_rank: bool = False

    @torch.no_grad()  # NOTE: No gradients needed
    def compute_abundances(
        self,
        Y,
        D,
        r,
        *args,
        **kwargs,
    ):
        # GPU access
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        # Problem dimensions
        p, n = Y.shape
        m = D.shape[1]

        def loss(a, b):
            return 0.5 * ((Y - (D @ b) @ a) ** 2).sum()

        # Timing
        tic = time.time()

        A = (1 / r) * torch.ones((r, n))
        B = (1 / m) * torch.ones((m, r))
        L1 = torch.zeros((r, n))
        L2 = torch.zeros((m, r))
        L3 = torch.zeros((p, r))
        S1 = L1
        S2 = L2
        S3 = L3
        Y = torch.Tensor(Y)
        D = torch.Tensor(D)
        # Send matrices on GPU
        D = D.to(device)
        Y = Y.to(device)
        A = A.to(device)
        B = B.to(device)
        S1 = S1.to(device)
        S2 = S2.to(device)
        S3 = S3.to(device)
        L1 = L1.to(device)
        L2 = L2.to(device)
        L3 = L3.to(device)
        eye_r = torch.eye(r).to(device)
        eye_m = torch.eye(m).to(device)
        ones_r = torch.ones(r).to(device)
        ones_m = torch.ones(m).to(device)
        ones_n = torch.ones(n).to(device)

        Q1inv = self.mu3 * D.t() @ D + self.mu2 * eye_m
        Z1 = torch.linalg.solve(Q1inv, ones_m)
        c1 = -1 / torch.dot(ones_m, Z1)

        U1 = eye_m + c1 * torch.outer(Z1, ones_m)
        V1 = c1 * torch.outer(Z1, ones_r)

        m = torch.mean(Y, 1)

        Initloss = loss(A, B)
        logs.info(f"Initial loss => {Initloss:.3e}")
        progress = tqdm(range(self.T))
        for ii in progress:
            updateloss = loss(A, B)
            progress.set_postfix_str(f"loss={updateloss:.4e}")

            Q2inv = self.mu3 * S3.t() @ S3 + self.mu1 * eye_r
            Z2 = torch.linalg.solve(Q2inv, ones_r)
            c2 = -1 / torch.dot(ones_r, Z2)
            U2 = eye_r + c2 * torch.outer(Z2, ones_r)
            V2 = c2 * torch.outer(Z2, ones_n)

            for jj in range(self.TA):
                WA = S3.t() @ Y + self.mu1 * (S1 - L1)
                A = U2 @ torch.linalg.solve(Q2inv, WA) - V2
                S1 = A + L1
                S1[S1 <= 0] = 0
                L1 = L1 + A - S1

            Q3inv = A @ A.t() + (self.mu3 + self.lamb) * eye_r

            for jj in range(self.TB):
                WB = self.mu3 * D.t() @ (S3 - L3) + self.mu2 * (S2 - L2)
                B = U1 @ torch.linalg.solve(Q1inv, WB) - V1
                S2 = B + L2
                S2[S2 <= 0] = 0
                S3 = torch.linalg.solve(
                    Q3inv,
                    Y @ A.t()
                    + self.lamb * torch.outer(m, ones_r)
                    + self.mu3 * (D @ B + L3),
                    left=False,
                )
                L2 = L2 + B - S2
                L3 = L3 + D @ B - S3

        tac = time.time()
        self.register_time(tac - tic)
        logs.info(self.processing_time)

        logs.info(f"Final loss => {loss(A, B):.2e}")
        A = A.cpu().numpy()
        B = B.cpu().numpy()
        # return A, B
        if self.low_rank:
            return A
        return B @ A  # full-rank abundances!


@dataclass
class SUnShrink(BaseUnmixingModel):
    mu1: float = 100.0
    mu2: float = 10.0
    mu3: float = 1.0
    TA: int = 5
    TB: int = 5
    T: int = 5000
    lambd: float = 0.1
    hard: bool = True
    low_rank: bool = False

    @property
    def shrink(self):
        return (
            torch.nn.Softshrink(self.lambd / self.mu2)
            if not self.hard
            else torch.nn.Hardshrink(sqrt(2 * self.lambd / self.mu2))
        )

    @torch.no_grad()  # NOTE: No gradients needed
    def compute_abundances(
        self,
        Y,
        D,
        r,
        *args,
        **kwargs,
    ):
        # GPU access
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        # Print information about shrinkage type
        logs.info(f"Using hard thresholding? {self.hard}")

        # Problem dimensions
        p, n = Y.shape
        m = D.shape[1]

        def loss(a, b):
            penalty = (
                self.lambd * b.abs().sum()
                if not self.hard
                else self.lambd * b[b.abs() < EPS].sum()
            )
            return 0.5 * ((Y - (D @ b) @ a) ** 2).sum() + penalty

        # Timing
        tic = time.time()

        A = (1 / r) * torch.ones((r, n))
        # B = (1 / m) * torch.ones((m, r))
        B = torch.zeros((m, r))
        L1 = torch.zeros((r, n))
        L2 = torch.zeros((m, r))
        L3 = torch.zeros((p, r))
        S1 = L1
        S2 = L2
        S3 = L3
        Y = torch.Tensor(Y)
        D = torch.Tensor(D)
        # Send matrices on GPU
        D = D.to(device)
        Y = Y.to(device)
        A = A.to(device)
        B = B.to(device)
        S1 = S1.to(device)
        S2 = S2.to(device)
        S3 = S3.to(device)
        L1 = L1.to(device)
        L2 = L2.to(device)
        L3 = L3.to(device)
        eye_r = torch.eye(r).to(device)
        eye_m = torch.eye(m).to(device)
        ones_r = torch.ones(r).to(device)
        ones_n = torch.ones(n).to(device)

        Q1inv = self.mu3 * D.t() @ D + self.mu2 * eye_m

        Initloss = loss(A, B)
        logs.info(f"Initial loss => {Initloss:.3e}")
        progress = tqdm(range(self.T))
        for ii in progress:
            updateloss = loss(A, B)
            progress.set_postfix_str(f"loss={updateloss:.4e}")

            Q2inv = self.mu3 * S3.t() @ S3 + self.mu1 * eye_r
            Z2 = torch.linalg.solve(Q2inv, ones_r)
            c2 = -1 / torch.dot(ones_r, Z2)
            U2 = eye_r + c2 * torch.outer(Z2, ones_r)
            V2 = c2 * torch.outer(Z2, ones_n)

            for jj in range(self.TA):
                WA = S3.t() @ Y + self.mu1 * (S1 - L1)
                A = U2 @ torch.linalg.solve(Q2inv, WA) - V2
                S1 = A + L1
                S1[S1 <= 0] = 0
                L1 = L1 + A - S1

            Q3inv = A @ A.t() + self.mu3 * eye_r

            for jj in range(self.TB):
                WB = self.mu3 * D.t() @ (S3 - L3) + self.mu2 * (S2 - L2)
                # B = U1 @ torch.linalg.solve(Q1inv, WB) - V1
                B = torch.linalg.solve(Q1inv, WB)
                S2 = self.shrink(B + L2)
                S2[S2 <= 0] = 0
                # TODO: upper bound?
                S2[S2 >= 1] = 1
                S3 = torch.linalg.solve(
                    Q3inv, Y @ A.t() + self.mu3 * (D @ B + L3), left=False
                )
                L2 = L2 + B - S2
                L3 = L3 + D @ B - S3

        tac = time.time()
        self.register_time(tac - tic)
        logs.info(self.processing_time)

        logs.info(f"Final loss => {loss(A, B):.2e}")
        A = A.cpu().numpy()
        B = S2.cpu().numpy()
        # return A, B
        if self.low_rank:
            return A
        return B @ A  # full-rank abundances!
