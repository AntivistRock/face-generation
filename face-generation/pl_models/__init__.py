from __future__ import annotations
from typing import Literal, TypedDict, TYPE_CHECKING
import hydra


if TYPE_CHECKING:
    from typing_extensions import NotRequired


@hydra.main()
def create_speed_estimator(config: ORBConfig | NeuFlowConfig | ECOTRConfig):
    if config["name"] == "orb":
        from vae import 
    elif config["name"] == "sift":
        from .key_points import KeyPoints

        return KeyPoints(kp_type = 'sift')
    elif config["name"] == "surf":
        from .key_points import KeyPoints

        return KeyPoints(kp_type = 'surf')
    elif config["name"] == "neuflow":
        from .neuflow import NeuFlow

        return NeuFlow()
    elif config["name"] == 'ecotr':
        from .ecotr import EcoTr

        return EcoTr()
