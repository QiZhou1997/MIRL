import numpy as np
import warnings
from collections import OrderedDict

from mirl.pools.utils import get_batch, _random_batch_independently, _shuffer_and_random_batch
from mirl.pools.simple_pool import SimplePool
import mirl.torch_modules.utils as ptu
import copy

class FramePool(SimplePool):
    # save observation with np.uint8
    def _get_default_fields(self):
        o_shape = self._observation_shape 
        a_shape = self._action_shape 
        return {
            'observations': {
                'shape': o_shape,
                'type': np.uint8,
            },
            'next_observations': {
                'shape': o_shape,
                'type': np.uint8,
            },
            'actions': {
                'shape': a_shape,
                'type': np.float32,
            },
            'rewards': {
                'shape': (1,),
                'type': np.float32,
            },
            'terminals': {
                'shape': (1,),
                'type': np.float32,
            },
        }

    def random_batch_torch(self, batch_size):
        batch = self.random_batch(batch_size)
        batch = ptu.np_to_pytorch_batch(batch)
        return batch