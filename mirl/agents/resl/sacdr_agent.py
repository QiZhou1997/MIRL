from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim 

import mirl.torch_modules.utils as ptu
from mirl.agents.sac_agent import SACAgent
from mirl.utils.eval_util import create_stats_ordered_dict
from mirl.utils.logger import logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import copy

def plot_scatter(x, y, name, n_step):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.grid(ls='--')
    ax.scatter(x, y, alpha=0.2)
    logger.tb_add_figure(name, fig, n_step)

# TODO algorithm get_item替代getattr __dict__
class SACDRAgent(SACAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        pool,
        normalize_obs=True,
        reg_value=True,
        norm_q=True,
        momentum=5e-2,
        beta=1,
        **sac_kwargs
    ):
        super().__init__(
            env,
            policy,
            qf,
            qf_target,
            **sac_kwargs
        )
        self.da = self.env.action_space.shape[0]
        self.ds = self.env.observation_space.shape[0]
        self.norm_q = norm_q
        self.reg_value = reg_value
        self.normalize_obs = normalize_obs
        self.beta = beta
        self.momentum = momentum
        self.q_std = ptu.ones((1,))
        if normalize_obs:
            assert pool is not None
            assert pool.compute_mean_std
            self.obs_mean, self.obs_std = pool.get_mean_std()['observations']
            self.obs_mean = ptu.from_numpy(self.obs_mean)
            self.obs_std = ptu.from_numpy(self.obs_std)
            self.k = pool.k

    def process_obs(self, obs):
        if self.normalize_obs:
            obs = obs = (obs-self.obs_mean)/(self.obs_std+1e-6)
        return obs

    def step_explore(self, o, **kwargs):
        raise NotImplementedError

    def step_exploit(self, o, **kwargs):
        with self.policy.deterministic_(True):
            o = ptu.from_numpy(o)
            o = self.process_obs(o)
            a,_ = self.policy.action(o, **kwargs)
            a = ptu.get_numpy(a)
        return a, {}

    def _get_distance(self, a, k_actions, k_distance, norm='l2'):
        a_dist = a[:,None,:] - k_actions
        if norm == 'l2':
            k_distance = k_distance**2/self.ds
            a_dist = (a_dist**2).mean(dim=-1)
            dist = a_dist+k_distance
        if norm == 'l1':
            k_distance = k_distance/self.ds
            a_dist = torch.abs(a_dist).mean(dim=-1)
            dist = a_dist+k_distance
        dist,ind = torch.min(dist, dim=-1, keepdim=True)
        with torch.no_grad():
            s_dist = k_distance.gather(1, ind)
            a_dist = a_dist.gather(1, ind)
        return dist, s_dist, a_dist, ind

    def log_dr(self, dist, s_dist, a_dist, ind):
        if self._log_tb_or_not():
            logger.tb_add_histogram('dr/dist_hist', dist, self.num_train_steps)
            logger.tb_add_histogram('dr/s_dist_hist', s_dist, self.num_train_steps)
            logger.tb_add_histogram('dr/a_dist_hist', a_dist, self.num_train_steps)
            logger.tb_add_scalar('dr/dist', dist.mean(), self.num_train_steps)
            logger.tb_add_scalar('dr/s_dist', s_dist.mean(), self.num_train_steps)
            logger.tb_add_scalar('dr/a_dist', a_dist.mean(), self.num_train_steps)
            plot_scatter(ind, dist, 'dr/ind_vs_dist', self.num_train_steps)
            plot_scatter(ind, s_dist, 'dr/ind_vs_sdist', self.num_train_steps)
            plot_scatter(ind, a_dist, 'dr/ind_vs_adist', self.num_train_steps)
            count = []
            for i in range(self.k):
                n = (ind == i).sum()
                n = n.item()
                count.append(n)
            fig, ax = plt.subplots()
            ax.pie(count, labels=list(np.arange(self.k)), autopct='%1.0f%%',
                    shadow=False, startangle=170)
            logger.tb_add_figure('ind_pie', fig, self.num_train_steps)

    def compute_q_target(self, next_obs, rewards, terminals, 
            next_ka, next_dist, v_pi_kwargs={}):
        with torch.no_grad():
            next_action, _ = self.policy.action(
                next_obs, **self.next_sample_kwargs
            )
            target_q_next_action = self.qf_target.value(
                next_obs, 
                next_action, 
                **v_pi_kwargs
            )[0]
            if self.reg_value:
                if self.norm_q:
                    beta = self.beta
                else:
                    beta = self.beta
                dist,_,_,_ = self._get_distance(next_action, next_ka, next_dist)
                target_q_next_action = target_q_next_action - beta*dist
            q_target = rewards + (1.-terminals)*self.discount*target_q_next_action
        return q_target

    def compute_policy_loss(self, obs, new_action, action_info, dist, v_pi_kwargs={}):
        log_prob_new_action = action_info['log_prob']
        alpha = self._get_alpha().detach()
        q_new_action, _ = self.qf.value(
            obs, 
            new_action, 
            return_ensemble=False, 
            **v_pi_kwargs
        )
        entropy = -log_prob_new_action.mean()
        dr = dist.mean()
        q_pi_mean = q_new_action.mean()
        if self.norm_q:
            with torch.no_grad():
                std = q_new_action.std()
                lr = self.momentum
                std = (self.q_std*lr + std*(1-lr)).detach()
                self.q_std = std
        else:
            std = 1
        policy_loss = -alpha*entropy + self.beta*dr - q_pi_mean/std
        policy_info = self._log_policy_info(
            new_action, action_info, policy_loss,
            q_pi_mean, entropy)
        if self._log_tb_or_not(): 
            logger.tb_add_histogram('dist', dist, self.num_train_steps)
            logger.tb_flush()
        policy_info['policy/dr'] = dr.item()
        return policy_loss, policy_info


    def train_from_torch_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        k_actions = batch['k_actions']
        obs_distance = batch['obs_distance']
        next_k_actions = batch['next_k_actions']
        next_obs_distance = batch['next_obs_distance']
        obs = self.process_obs(obs)
        next_obs = self.process_obs(next_obs)
        self.log_batch(rewards, terminals)
        #################
        # update critic #
        #################
        q_target = self.compute_q_target(
            next_obs, rewards, terminals, 
            next_k_actions, next_obs_distance, self.next_v_pi_kwargs)
        qf_loss, train_qf_info = self.compute_qf_loss(obs, actions, q_target)
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.log_critic_grad_norm()
        self.qf_optimizer.step()
        if self.num_train_steps % self.target_update_freq == 0:
            self._update_target(self.soft_target_tau)
        self.train_info.update(train_qf_info)
        ###########################
        # update actor and alpha #
        ###########################
        if self.num_train_steps % self.policy_update_freq == 0:
            new_action, action_info = self.policy.action(
                obs, **self.current_sample_kwargs
            )
            dist, s_dist, a_dist, ind = self._get_distance(new_action, k_actions, obs_distance)
            self.log_dr(dist, s_dist, a_dist, ind)
            #update alpha
            if self.use_automatic_entropy_tuning:
                alpha_loss, train_alpha_info = self.compute_alpha_loss(action_info)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.train_info.update(train_alpha_info)
            #update policy
            policy_loss, train_policy_info = self.compute_policy_loss(
                obs, 
                new_action, 
                action_info, 
                dist,
                self.current_v_pi_kwargs
            )
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            self.train_info.update(train_policy_info)
        #####################
        # update statistics #
        #####################
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics.update(self.train_info)
        self.log_train_info()
        self.num_train_steps += 1
        return copy.deepcopy(self.train_info)
