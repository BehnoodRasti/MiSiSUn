# Imports
from logging import getLogger
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import scipy.io as sio
import numpy as np

# Public concepts
__all__ = ["HSImage"]

# Logs
logs = getLogger(__name__)

# Global variables
INTEGER_FIELDS = (
    "h",
    "w",
    "m",
    "p",
    "r",
    "n",
)
EPS = 1e-10


@dataclass
class HSImage:
    # Metadata
    name: str
    # Paths
    data_dir: str = "./data"

    def __post_init__(self) -> None:
        filename = f"{self.name}.mat"
        path = Path(self.data_dir, filename)
        logs.debug(f"Path to be opened: {path}")
        assert path.is_file()
        self.path = path
        self._load()

    def _load(self) -> None:
        data = sio.loadmat(self.path)
        logs.debug(f"Data keys: {data.keys()}")

        for key in filter(
            lambda k: not k.startswith("__"),
            data.keys(),
        ):
            self.__setattr__(
                key, data[key].item() if key in INTEGER_FIELDS else data[key]
            )

        if "n" not in data.keys():
            self.n = self.h * self.w

        # Check data
        assert self.n == self.h * self.w
        assert self.Y.shape == (self.p, self.n)  # 2D image

        self.has_dict = False
        if "D" in data.keys():
            self.has_dict = True
            assert self.D.shape == (self.p, self.m)

        if "index" in data.keys():
            self.index = list(self.index.squeeze())

        self.has_GT = False
        if "A" in data.keys():
            self.has_GT = True
            assert self.E.shape == (self.p, self.r)
            assert self.A.shape == (self.r, self.n)

            # Check physical constraints
            # Abundance Sum-to-One constraint (ASC)
            assert np.allclose(
                self.A.sum(0),
                np.ones(self.n),
                rtol=1e-3,
                atol=1e-3,
            )
            # Abundance non-negative constraint (ANC)
            assert np.all(self.A >= -EPS)
            # Endmembers non-negative constraint (ENC)
            assert np.all(self.E >= -EPS)

        self.has_labels = False
        if "labels" in data.keys():
            self.has_labels = True
            try:
                assert len(self.labels) == self.r
                tmp_labels = list(self.labels)
                self.labels = [s.strip(" ") for s in tmp_labels]
            except Exception:
                # Create numeroted labels
                self.labels = [f"#{ii}" for ii in range(self.r)]

    def __call__(self) -> Tuple[np.ndarray, int, np.ndarray]:
        """Get hyperspectral image data."""
        return (self.Y, self.r, self.D)

    def get_GT(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get endmembers and abundances GT."""
        if not self.has_GT:
            raise ValueError("No GT found for this dataset...")
        return (self.E, self.A)

    def get_dimensions(self) -> dict:
        return {
            "bands": self.p,
            "pixels": self.n,
            "lines": self.h,
            "samples": self.w,
            "atoms": self.m,
        }

    def get_shape(self) -> Tuple[int, int]:
        return (self.h, self.w)

    def get_labels(self) -> list:
        return self.labels

    def get_index(self) -> list:
        return self.index

    def __repr__(self) -> str:
        msg = f"HSI => {self.name}\n"
        msg += "-------------------------------\n"
        for key, value in self.get_dimensions().items():
            msg += f"{key}: {value}\n"
        msg += f"GlobalMinValue: {self.Y.min()}\n"
        msg += f"GlobalMaxValue: {self.Y.max()}\n"
        return msg


if __name__ == "__main__":
    hsi = HSImage("sim1")
    data = hsi()
    print(hsi)
