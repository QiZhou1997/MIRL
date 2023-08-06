import gym
import warnings

from mirl.pools.base_pool import Pool
from mirl.utils.logger import logger

class OfflinePool(Pool):
    def __init__(
        self, 
        dataset_name,
        pool_class="SimplePool", 
        **pool_kwargs
    ):
        import d4rl
        env = gym.make(dataset_name)
        pool_kwargs['env'] = env
        d = self.origin_dataset = d4rl.qlearning_dataset(env)
        size = len(d['rewards'])
        pool_kwargs['max_size'] = size
        from mirl.utils.initialize_utils import get_item
        self.inner_pool = get_item('pool', pool_class, pool_kwargs)
        d['rewards'] = d['rewards'][:,None]
        d['terminals'] = d['terminals'][:,None]
        self.inner_pool.add_samples(d)
    
    def __getattribute__(self, name):
        if name == "inner_pool":
            return object.__getattribute__(self, "inner_pool")
        elif name == "origin_dataset":
            return object.__getattribute__(self, "origin_dataset")
        else:
            return self.inner_pool.__getattribute__(name)



        