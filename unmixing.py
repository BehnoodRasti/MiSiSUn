from typing import Type
from logging import getLogger

from src.noise import AdditiveWhiteGaussianNoise
from src.data import HSImage
from src.utils import SVD_projection
from src.model.base import BaseUnmixingModel
from src.model.sunsal import SUnSAL, CLSUnSAL, S2WSU
from src.model.mua import MUA_SLIC
from src.model.deep_image_prior import SUnCNN
from src.model.archetypal_analysis import SUnAA, FaSUn, MiSiSUn, SUnShrink
from src.metrics import compute_metric, SRE, aRMSE

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
Cuprite_HSI = HSImageConf(name="Cuprite")

# Register image configs under group: image
hsi_store = store(group="hsi")

hsi_store(Sim1_HSI, name="sim1")
hsi_store(MR_rho100_HSI, name="rho100")
hsi_store(MR_rho85_HSI, name="rho85")
hsi_store(MR_rho70_HSI, name="rho70")
hsi_store(Cuprite_HSI, name="cuprite")

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


# Register model config under group: model
model_store = store(group="model")

model_store(SUnSAL, name="SUnSAL")
model_store(CLSUnSAL, name="CLSUnSAL")
model_store(S2WSU, name="S2WSU")
model_store(MUA_SLIC, name="MUA_SLIC")
model_store(SUnCNN, name="SUnCNN")
model_store(SUnAA, name="SUnAA")
model_store(FaSUn, name="FaSUn")
model_store(SUnShrink, name="SUnS")
model_store(MiSiSUn, name="MiSiSUn")


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
        {
            "model": "SUnSAL",
        },
    ],
)
def unmix(
    noise: Type[AdditiveWhiteGaussianNoise],
    hsi: Type[HSImage],
    model: Type[BaseUnmixingModel],
    l2_normalize: bool = False,
    SVD_project: bool = False,
):
    logs.info("SEMI-SUPERVISED UNMIXING...[START]")
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
    # model
    # NOTE: Return full abundances
    A = model.compute_abundances(Y, D, r=r, h=h, w=w)

    if hsi.has_GT:
        # Get abundances ground truth
        _, A_GT = hsi.get_GT()
        # A_GT.shape => (r, n)
        # Get index
        A1 = A[hsi.get_index()]
        # Get labels
        labels = hsi.get_labels()
        # Compute SRE
        logs.info(
            compute_metric(
                SRE(),
                A_GT,
                A1,
                labels,
                detail=False,
                on_endmembers=False,
            )
        )
        # Compute aRMSE
        logs.info(
            compute_metric(
                aRMSE(),
                A_GT,
                A1,
                labels,
                detail=True,
                on_endmembers=False,
            )
        )

    logs.info("SEMI-SUPERVISED UNMIXING...[END]")


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(unmix).hydra_main(
        config_name="unmixing",
        version_base="1.3",
        config_path=".",
    )
