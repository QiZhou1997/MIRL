from mirl.policies.base_policy import Policy
import numpy as np
from torch import nn
import torch

def make_determinsitic(random_policy):
    return MakeDeterministic(random_policy)

class MakeDeterministic(nn.Module, Policy):
    def __init__(self, random_policy):
        nn.Module.__init__(self)
        self.random_policy = random_policy

    def action(self, obs, **kwargs):
        with self.random_policy.deterministic_(True):
            return self.random_policy.action(obs, **kwargs)

    def action_np(self, obs, **kwargs):
        with self.random_policy.deterministic_(True):
            return self.random_policy.action_np(obs, **kwargs)

    def reset(self, **kwarg):
        self.random_policy.reset(**kwarg)

    def save(self, **kwarg):
        self.random_policy.save(**kwarg)

    def load(self, **kwarg):
        self.random_policy.load(**kwarg)
