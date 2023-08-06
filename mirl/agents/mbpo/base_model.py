import torch
from torch import nn
import mirl.torch_modules.utils as ptu
from mirl.processors.normalizer import Normalizer
from mirl.components import Component

class Model(Component, nn.Module):
    def __init__(
        self, 
        env,
        normalize_obs=True,
        normalize_action=True,
        normalize_delta=True,
        normalize_reward=False,
        known=['done']
    ):
        nn.Module.__init__(self)
        self.env = env
        self.env_name = env.env_name
        self.need_to_learn = []
        # need to learn what
        from mirl.environments.reward_done_functions import get_reward_done_function
        self.reward_f, self.done_f = get_reward_done_function(self.env_name, known)
        self.observation_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape
        self.processed_obs_shape = self.observation_shape
        self.processed_action_shape = self.action_shape
        if 'reward' in known:
            self.learn_reward = False
        else:
            self.learn_reward = True
            self.need_to_learn.append('reward')
        if 'done' in known:
            self.learn_done = False
        else:
            self.learn_done = True
            self.need_to_learn.append('done')
        self.normalize_obs = normalize_obs
        self.normalize_action = normalize_action
        self.normalize_delta = normalize_delta
        self.normalize_reward = normalize_reward
        if normalize_obs:
            self.obs_processor = Normalizer(self.observation_shape)
        if normalize_action:
            self.action_processor = Normalizer(self.action_shape)
        if normalize_delta:
            self.delta_processor = Normalizer(self.observation_shape)
        if self.learn_reward and normalize_reward:
            self.reward_processor = Normalizer((1,))
        
    def _predict_delta(self, obs, action, **kwargs):
        raise NotImplementedError

    def step(self, obs, action, **kwargs):
        # process input
        if self.normalize_obs:
            processed_obs = self.obs_processor.process(obs)
        else:
            processed_obs = obs
        if self.normalize_action:
            processed_action = self.action_processor.process(action)
        else:
            processed_action = action
        # predict
        delta, reward, done, info = self._predict_delta(processed_obs, processed_action, **kwargs)
        # recover output
        if self.normalize_delta:
            delta = self.delta_processor.recover(delta)
        next_obs = obs + delta
        if not self.learn_reward:
            reward = self.reward_f(obs, action, next_obs, "torch")
        elif self.learn_reward and self.normalize_reward:
            reward = self.reward_processor.recover(reward)
        if not self.learn_done:
            done = self.done_f(obs, action, next_obs, "torch")
        return next_obs, reward, done, info

    def step_np(self, obs, action, **kwargs):
        obs = ptu.from_numpy(obs)
        action = ptu.from_numpy(action)
        next_obs, reward, done, info = self.step(obs, action, **kwargs)
        next_obs = ptu.get_numpy(next_obs)
        reward = ptu.get_numpy(reward)
        done = ptu.get_numpy(done)
        info = ptu.torch_to_np_info(info)
        return next_obs, reward, done, info

    def rollout_given_actions(self, actions):
        raise NotImplementedError

    # TODO check
    def rollout_given_policy(
        self, o, agent, n_step, 
        agent_step_policy={}, 
        env_step_kwargs={}
    ):
        paths = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': []
        }
        for _ in range(n_step):
            a = agent.step(**agent_step_policy)
            o, r, d, _ = self.step(o, a, **env_step_kwargs)
            paths['observations'].append(o)
            paths['actions'].append(a)
            paths['rewards'].append(r)
            paths['terminals'].append(d)
        return paths

    def rollout_given_policy_np(
        self, o, agent, n_step, 
        agent_step_policy={}, 
        env_step_kwargs={}
    ):
        o = ptu.from_numpy(o)
        new_paths = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': []
        }
        paths = self.rollout_given_policy(
            o, agent, n_step, 
            agent_step_policy, 
            env_step_kwargs
        )
        for k in paths:
            for item in paths[k]:
                new_paths[k].append(ptu.from_numpy(item))
        return new_paths

    def reset(self):
        pass 

    def save(self, save_dir=None):
        pass
    
    def load(self, load_dir=None):
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}
