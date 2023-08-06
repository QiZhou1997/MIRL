from mirl.values.ensemble_value import EnsembleQValue
from mirl.values.base_value import QValue
from mirl.torch_modules.mlp import MLP
from torch import nn
import torch
import mirl.torch_modules.utils as ptu

class LatentEnsembleQValue(EnsembleQValue): 
    def __init__( 
        self, 
        env, 
        processor,
        value_name='latent_q_value',
        **ensemble_q_kwargs
    ):
        self.feature_size = processor.output_shape[0] + env.action_space.shape[0]
        super().__init__(env, value_name=value_name, **ensemble_q_kwargs)

    def _get_feature_size(self):
        return self.feature_size

