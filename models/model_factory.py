# Warsaw University of Technology

import torch.nn as nn

from models.minkloc import MinkLoc
from misc.utils import ModelParams
from models.layers.pooling_wrapper import PoolingWrapper
from models.minkbev import MinkBEVBackbone


def model_factory(model_params: ModelParams):
    """
    模型工厂（BEV专用版本）
    """
    if model_params.model == 'MinkLocBEV':
        in_channels = getattr(model_params, 'in_channels', 32)

        print(f"Model Factory: Initializing MinkLocBEV...")
        print(f"  Input Channels (Z-layers): {in_channels}")
        print(f"  Feature Size (Backbone Out): {model_params.feature_size}")

        backbone = MinkBEVBackbone(in_channels=in_channels,
                                   out_channels=model_params.feature_size,
                                   dimension=2)

        pooling = PoolingWrapper(pool_method=model_params.pooling,
                                 in_dim=model_params.feature_size,
                                 output_dim=model_params.output_dim)

        model = MinkLoc(backbone=backbone, pooling=pooling,
                       normalize_embeddings=model_params.normalize_embeddings)

    elif model_params.model == 'MinkLoc':
        raise NotImplementedError(
            "MinkLoc (3D) has been removed from this codebase. "
            "Please use model=MinkLocBEV in your config file."
        )
    else:
        raise NotImplementedError(f'Model not implemented: {model_params.model}')

    return model