from typing import Type
from logging import getLogger

# from src.configs import UnmixingConfig
from src.noise import AdditiveWhiteGaussianNoise
from src.data import HSImage
from src.utils import SVD_projection
from src.model import BatchVCA

from hydra_zen import zen, store, make_custom_builds_fn
import numpy as np

logs = getLogger(__name__)

builds = make_custom_builds_fn(populate_full_signature=True)

# Create HSI configs
HSImageConf = builds(HSImage)
Sim1_HSI = HSImageConf(name="sim1")
MR_rho100_HSI = HSImageConf(name="MR_rho100_N10000")
MR_rho85_HSI = HSImageConf(name="MR_rho85_N10000")
MR_rho70_HSI = HSImageConf(name="MR_rho70_N10000")

# Register image configs under group: image
hsi_store = store(group="hsi")

hsi_store(Sim1_HSI, name="sim1")
hsi_store(MR_rho100_HSI, name="rho100")
hsi_store(MR_rho85_HSI, name="rho85")
hsi_store(MR_rho70_HSI, name="rho70")

# Create noise configs
NoiseConf = builds(AdditiveWhiteGaussianNoise)
Noiseless = NoiseConf(SNR=0.0)  # NOTE: 0.0 => convention for lack of added noise!
AWGN_20dB = NoiseConf(SNR=20.0)
AWGN_30dB = NoiseConf(SNR=30.0)
AWGN_40dB = NoiseConf(SNR=40.0)

# Register noise configs under group: noise
noise_store = store(group="noise")

noise_store(Noiseless, name="noiseless")
noise_store(AWGN_20dB, name="20dB")
noise_store(AWGN_30dB, name="30dB")
noise_store(AWGN_40dB, name="40dB")


# TODO: Create unmixing model config
# TODO: Register model config under group: model


@store(
    name="unmixing",
    hydra_defaults=[
        "_self_",
        {
            "hsi": "sim1",
        },
        {
            "noise": "30dB",
        },
    ],
)
def unmix(
    noise: Type[AdditiveWhiteGaussianNoise],
    hsi: Type[HSImage],
    l2_normalize: bool = False,
    SVD_project: bool = False,
):
    logs.info(hsi)
    # Get data
    Y, r, D = hsi()
    # Get image shape
    h, w = hsi.get_shape()
    # Apply noise
    Y = noise.noisify(Y)
    # L2 normalization
    if l2_normalize:
        Y /= np.linalg.norm(
            Y,
            axis=0,
            ord=2,
            keepdims=True,
        )
    # SVD projection
    if SVD_project:
        Y = SVD_projection(Y, r)


if __name__ == "__main__":
    # from hydra_zen import ZenStore

    # store = ZenStore(deferred_hydra_store=False)
    # store(UnmixingConfig, name="unmixing")
    store.add_to_hydra_store()
    zen(unmix).hydra_main(
        config_name="unmixing",
        version_base="1.3",
        config_path=".",
    )
