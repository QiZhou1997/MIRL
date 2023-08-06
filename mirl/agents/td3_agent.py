from mirl.agents.ddpg_agent import DDPGAgent
from collections import OrderedDict
import torch

class TD3Agent(DDPGAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        exploration_noise=0.1,
        policy_update_freq=2,
        next_sample_kwargs={
            'deterministic': False,
            'use_noise_clip': True
        },
        current_sample_kwargs={'deterministic': True},
        **ddpg_kwargs
    ):
        super().__init__(
            env,
            policy,
            qf,
            qf_target,
            policy_update_freq=policy_update_freq,
            next_sample_kwargs=next_sample_kwargs,
            current_sample_kwargs=current_sample_kwargs,
            **ddpg_kwargs
        )
        self.exploration_noise = exploration_noise

    def step_explore(self, o, **kwargs):
        with self.policy.deterministic_(False):
            with self.policy.noise_scale_(self.exploration_noise):
                return self.policy.action_np(o, **kwargs)
