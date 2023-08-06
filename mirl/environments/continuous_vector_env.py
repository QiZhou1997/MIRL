import random
import numpy as np
import gym
from gym.spaces import Box
from gym import Wrapper
from gym.vector import SyncVectorEnv, AsyncVectorEnv
import random
from mirl.environments.base_env import Env
from mirl.environments.utils import get_make_fns
from mirl.utils.mean_std import RunningMeanStd
from mirl.utils.logger import logger
from collections import Iterable
import warnings

class ContinuousVectorEnv(Env, Wrapper):
    def __init__(
        self,   
        env_name,
        n_env=1, 
        reward_scale=1.0,
        episode_length=np.inf,
        should_normalize_obs=False, 
        transform_action_space=False, 
        asynchronous=True,
        **vector_env_kwargs
    ):
        super().__init__(env_name)
        self._n_env = n_env
        self.cur_seeds = [random.randint(0,65535) for i in range(n_env)]
        self.make_fns = get_make_fns(env_name, self.cur_seeds, n_env)
        if asynchronous:
            inner_env = AsyncVectorEnv(self.make_fns,**vector_env_kwargs)
        else:
            inner_env = SyncVectorEnv(self.make_fns,**vector_env_kwargs)
        Wrapper.__init__(self, inner_env)
        self.asynchronous = asynchronous

        self.should_normalize_obs = should_normalize_obs
        self.observation_space = self.env.single_observation_space
        if should_normalize_obs:
            self.obs_mean_std = RunningMeanStd(self.observation_space.shape)

        self.transform_action_space = transform_action_space
        if transform_action_space:
            self.low = np.maximum(self.env.single_action_space.low, -10)
            self.high = np.minimum(self.env.single_action_space.high, 10)
            ub = np.ones(self.env.single_action_space.shape, dtype=np.float32)
            self.action_space = Box(-1.0* ub, 1.0*ub)
        else:
            self.action_space = self.env.single_action_space

        self.reward_scale = reward_scale
        self.episode_length = episode_length
        self.reset()

    @property
    def n_env(self):
        return self._n_env
    
    def _normalize_observation(self, obs):
        return (obs - self.obs_mean_std.mean) / np.sqrt(self.obs_mean_std.var + 1e-12)

    def get_state(self):
        assert not self.asynchronous
        sim_data = []
        for e in self.env.envs:
            sim_data.append(e.sim.get_state())
        return sim_data

    def set_state(self, sim_data):
        assert not self.asynchronous
        if isinstance(sim_data,Iterable):
            for e,d in zip(self.env.envs, sim_data):
                e.sim.set_state(d)
        else:
            for e in self.env.envs:
                e.sim.set_state(sim_data)

    def state_vector(self):
        assert not self.asynchronous
        obs = []
        for e in self.env.envs:
            if hasattr(e, '_get_obs'):
                obs.append(e._get_obs())
            else:
                obs.append(e.state_vector())
        return np.stack(obs)

    def reset(self):
        self.cur_step_id = 0
        obs = self.env.reset()
        if self.should_normalize_obs:
            self.obs_mean_std.update(obs)
            obs = self._normalize_observation(obs)
        return obs

    def step(self, action):
        self.cur_step_id = self.cur_step_id + 1
        if self.transform_action_space:
            action = np.clip(action, -1.0, 1.0)
            action = self.low + (action + 1.0) * (self.high - self.low) * 0.5
        if len(action.shape) == len(self.action_space.shape):
            action = np.stack([action] * self.n_env)
        o,r,d,infos=self.env.step(action)
        if self.should_normalize_obs:
            self.obs_mean_std.update(o)
            o = self._normalize_observation(o)
        r = r.reshape(self.n_env,1)
        d = d.reshape(self.n_env,1).astype(np.float32)
        if logger.log_or_not(logger.WARNING):
            new_info = {}
            for k in infos[0]:
                if k != "terminal_observation":
                    new_info[k] = np.empty((self.n_env,1),dtype=np.float32)
            for k,v in new_info.items():
                for i,info in enumerate(infos):
                    v[i,0] = info[k]
        else:
            new_info = {}
        return o, r, d, new_info

        # infos: list of dict
        

if __name__ == "__main__":
    e = ContinuousVectorEnv(
        "cheetah",
        n_env=10,
        should_normalize_obs=False, 
        transform_action_space=True, 
    )
    o = e.reset()
    for i in range(1000):
        next_o, r, d, info = e.step(e.action_space.sample())
        