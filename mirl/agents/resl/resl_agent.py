from mirl.agents.ddpg_agent import DDPGAgent
from mirl.agents.base_agent import Agent
from collections import OrderedDict
import torch.optim as optim 
import numpy as np
import torch
import mirl.torch_modules.utils as ptu
import torch.nn.functional as F
import copy
from mirl.utils.logger import logger


class KNearestAgent(DDPGAgent): 
    def __init__(
        self,
        env,
        qf,
        qf_target,
        pool,
        normalize_obs=True,
        dist_weight=1000,
        temp=0.1,
        num_seed_steps=-1,
        discount=0.99,
        qf_lr=3e-4,
        soft_target_tau=5e-3,
        policy_update_freq=1,
        target_update_freq=1,
        clip_error=None,
        optimizer_class='Adam',
        optimizer_kwargs={},
        next_sample_kwargs={},
        current_sample_kwargs={}, 
        next_v_pi_kwargs={}, 
        current_v_pi_kwargs={}, 
        tb_log_freq=100,
    ):
        Agent.__init__(self, env)
        if isinstance(optimizer_class, str):
            self.optimizer_class = eval('optim.'+optimizer_class)
        self.optimizer_kwargs = optimizer_kwargs
        self.qf = qf
        self.qf_target =qf_target
        self._update_target(1)
        self.num_train_steps = 0
        self.num_seed_steps = num_seed_steps
        self.soft_target_tau = soft_target_tau
        self.target_update_freq = target_update_freq
        self.policy_update_freq = policy_update_freq
        self.clip_error = clip_error
        self.qf_optimizer = self.optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
            **self.optimizer_kwargs
        )
        self.critic_params = list(self.qf.parameters())
        self.qf_lr = qf_lr
        self.discount = discount
        self.train_info = OrderedDict()
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True

        self.next_sample_kwargs = next_sample_kwargs
        self.current_sample_kwargs = current_sample_kwargs
        self.next_v_pi_kwargs = next_v_pi_kwargs
        self.current_v_pi_kwargs = current_v_pi_kwargs
        self.tb_log_freq = tb_log_freq
        if logger.log_or_not(logger.INFO):
            self.debug_info = {}

        self.normalize_obs = normalize_obs
        self.temp = temp
        self.dist_weight = 0-dist_weight
        if normalize_obs:
            self.pool = pool
            assert pool.compute_mean_std
            self.obs_mean, self.obs_std = pool.get_mean_std()['observations']
            self.obs_mean = ptu.from_numpy(self.obs_mean)
            self.obs_std = ptu.from_numpy(self.obs_std)

    def process_obs(self, obs):
        if self.normalize_obs:
            obs = obs = (obs-self.obs_mean)/(self.obs_std+1e-6)
        return obs
    
    def step_explore(self, o, **kwargs):
        raise NotImplementedError

    def step_exploit(self, o, **kwargs):
        np_k_actions, distance = self.pool.search(o, norm=True)
        k_actions = ptu.from_numpy(np_k_actions)
        o = ptu.from_numpy(o)
        distance = ptu.from_numpy(distance)
        o = self.process_obs(o)
        o_repeat = o[:,None,:].repeat(1,self.pool.k,1)
        q, _ = self.qf.value(o_repeat, k_actions)
        q = distance[...,None]*self.dist_weight+q
        max_ind = torch.argmax(q, dim=1)[:,0]
        max_ind = ptu.get_numpy(max_ind)
        arange = np.arange(max_ind.shape[0])
        a = np_k_actions[arange,max_ind]
        return a, {}

    def compute_q_target(self, next_obs, next_k_actions, next_k_distance,
                            rewards, terminals, v_pi_kwargs={}):
        with torch.no_grad():
            B, K, D = next_k_actions.shape
            next_obs_repeat = next_obs[:,None,:].repeat(1,K,1)
            next_q = self.qf_target.value(
                next_obs_repeat, 
                next_k_actions, 
                **v_pi_kwargs
            )[0]
            
            next_q = next_k_distance[...,None]*self.dist_weight+next_q
            prob = torch.softmax(next_q/self.temp, dim=1)
            next_q = (next_q*prob).sum(dim=1)
            q_target = rewards + (1.-terminals)*self.discount*next_q

            if self._log_tb_or_not():
                logger.tb_add_histogram('diff', diff[0], self.num_train_steps)
                logger.tb_add_histogram('q_value_ensemble', q_value_ensemble[0], self.num_train_steps)
                logger.tb_add_histogram('q_target', q_target, self.num_train_steps)
                logger.tb_flush()
        return q_target

    def train_from_torch_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        next_k_actions = batch['next_k_actions']
        next_obs_distance = batch['next_obs_distance']
        obs = self.process_obs(obs)
        next_obs = self.process_obs(next_obs)
        self.log_batch(rewards, terminals)
        ################
        # update critic #
        ################
        q_target = self.compute_q_target(next_obs, next_k_actions, next_obs_distance,
                        rewards, terminals, self.next_v_pi_kwargs)
        qf_loss, train_qf_info = self.compute_qf_loss(obs, actions, q_target)
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.log_critic_grad_norm()
        self.qf_optimizer.step()
        if self.num_train_steps % self.target_update_freq == 0:
            self._update_target(self.soft_target_tau)
        self.train_info.update(train_qf_info)
        #####################
        # update statistics #
        #####################
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics.update(self.train_info)
        self.log_train_info()
        self.num_train_steps += 1
        return copy.deepcopy(self.train_info)

    @property
    def networks(self):
        return [
            self.qf,
            self.qf_target,
        ]

    def get_snapshot(self):
        return dict(
            qf=self.qf,
            qf_target=self.qf_target,
        )