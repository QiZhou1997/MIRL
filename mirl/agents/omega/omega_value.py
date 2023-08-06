from sklearn import ensemble
from mirl.values.ensemble_value import EnsembleQValue
from mirl.values.base_value import QValue
from mirl.torch_modules.mlp import MLP
from torch.nn import Module
import numpy as np
import torch
from torch import nn


# class OmegaQValue(EnsembleQValue):
#     def __init__( 
#         self, 
#         env, 
#         ensemble_size=2,
#         value_name='ensemble_q_value',
#         **mlp_kwargs
#     ):
#         super().__init__(
#             env=env,
#             ensemble_size=ensemble_size,
#             value_name=value_name,
#             **mlp_kwargs
#         )
#         self.eval_module = MLP(
#             self._get_feature_size(), 
#             1,
#             **mlp_kwargs
#         )

#     def evaluate(self, obs, action):
#         input_tensor = self._get_features(obs, action)
#         return self.eval_module(input_tensor)

class OmegaQValue(nn.Module, QValue):
    def __init__( 
            self, 
            env, 
            ensemble_size=2,
            number_head=15, 
            value_name='omega_mh_q_value',
            **mlp_kwargs
        ):
        nn.Module.__init__(self)
        QValue.__init__(self, env)
        self.ensemble_size = ensemble_size
        assert self.ensemble_size is not None
        self.number_head = number_head
        self.module = MLP( 
            self._get_feature_size(), 
            number_head,
            ensemble_size=ensemble_size,
            module_name=value_name,
            **mlp_kwargs
        )
    
    def _get_feature_size(self):
        return self.observation_shape[0] + self.action_shape[0]

    def _get_features(self, obs, action):
        if obs.dim() > 2:
            obs = obs.unsqueeze(-3)
            action = action.unsqueeze(-3)
        return torch.cat([obs, action], dim=-1)

    def value(
        self, 
        obs, 
        action, 
        unc_penalty=0,
        use_min=False,
        only_return_ensemble=False,
        return_ensemble=True,
    ):
        input_tensor = self._get_features(obs, action)
        ensemble_values = self.module(input_tensor)
        info = {'ensemble_value': ensemble_values}
        if only_return_ensemble:
            return None, info
        ensemble_values = ensemble_values.transpose(0,1)
        shape = (ensemble_values.shape[0],-1)
        ensemble_values = ensemble_values.reshape(shape)
        if use_min:
            value = torch.min(ensemble_values, dim=-1, keepdim=True)[0]
            return value, info
        uncertainty = torch.std(ensemble_values, dim=-1, keepdim=True)
        info['uncertainty'] = uncertainty
        mean_value = torch.mean(ensemble_values, dim=-1, keepdim=True)
        value = mean_value - unc_penalty*uncertainty
        return value, info