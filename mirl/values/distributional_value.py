import torch
from torch import nn
import numpy as np
from mirl.torch_modules.mlp import MLP
import mirl.torch_modules.utils as ptu
from mirl.utils.logger import logger
from mirl.values.base_value import StateValue, QValue

# For TQC
class TQCQValue(nn.Module, QValue):
    def __init__( 
            self, 
            env, 
            ensemble_size=5,
            number_head=25, 
            value_name='tqc_q_value',
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

    def _drop_atoms(
        self, 
        value_atoms, 
        number_drop,
        percentage_drop,
    ):
        n_atom = value_atoms.shape[-1]
        # compute the number of dropped atoms
        if number_drop is None:
            if percentage_drop is None:
                number_drop = 0
            else:
                number_drop = round(n_atom*percentage_drop/100)

        # sort and then drop
        if number_drop > 0:
            value_atoms,_ = torch.sort(value_atoms, dim=-1) 
            value_atoms = value_atoms[...,:-number_drop]
            value = torch.mean(value_atoms,dim=-1,keepdim=True)

        elif number_drop == 0:
            value = torch.mean(value_atoms,dim=-1,keepdim=True)
        else:
            raise NotImplementedError

        return value, value_atoms

    def value(
        self, 
        obs, 
        action, 

        number_drop=None, 
        percentage_drop=None,

        return_atoms=False,
        return_ensemble=False,
        only_return_ensemble=False,
    ):
        input_tensor = self._get_features(obs, action)
        ensemble_atoms = self.module(input_tensor)

        if only_return_ensemble:
            info = {'ensemble_atoms': ensemble_atoms}
            return None, info

        ensemble_atoms = ensemble_atoms.transpose(-3,-2)
        shape = ensemble_atoms.shape[:-2] + (-1,)
        ensemble_atoms = ensemble_atoms.reshape(shape)

        value, final_value_atoms = self._drop_atoms(
            ensemble_atoms,
            number_drop,
            percentage_drop,
        )

        info = {}
        if return_atoms:
            info['value_atoms'] = final_value_atoms
        if return_ensemble:
            info['ensemble_atoms'] = ensemble_atoms
        return value, info

    def _get_feature_size(self):
        return self.observation_shape[0] + self.action_shape[0]

    def _get_features(self, obs, action):
        if obs.dim() > 2:
            obs = obs.unsqueeze(-3)
            action = action.unsqueeze(-3)
        return torch.cat([obs, action], dim=-1)