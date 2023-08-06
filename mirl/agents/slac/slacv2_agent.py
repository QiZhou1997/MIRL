from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.distributions import Normal

from mirl.utils.process import Progress, Silent, format_for_process
from mirl.utils.misc_untils import get_scheduled_value
import mirl.torch_modules.utils as ptu
from mirl.agents.slac.slac_agent import SLACAgent
from mirl.agents.drq.drq_agent import log_frame
from mirl.utils.logger import logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pdb


import numpy as np
import copy

#TODO： 用小的batch训练latent是否是因为出于计算效率/显存考虑？
#actor和critic的训练需要较大的batch，而序列梯度回传只能较小的batch？
#请测试前向不计算梯度和后向不计算梯度的耗时和显存开销
class SLACv2Agent(SLACAgent):
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        obs_processor,
        traj_processor,
        n_step_td=3,
        noise_schedule=[0,100000,1,0.1],
        **slac_kwargs
    ):
        sample_kwargs = {
            'deterministic': False,
            'use_noise_clip': True,
        }
        SLACAgent.__init__(
            self,
            env,
            policy,
            qf,
            qf_target,
            obs_processor,
            traj_processor,
            next_sample_kwargs=sample_kwargs,
            current_sample_kwargs=sample_kwargs,
            **slac_kwargs
        )
        del self.current_sample_kwargs['return_mean_std']
        self.n_step_td = n_step_td
        self.noise_schedule = noise_schedule
        self._true_discount = self.discount
        self.discount = self.discount ** n_step_td
    
    def set_noise_scale(self):
        noise_scale = get_scheduled_value(
            self.num_total_steps, 
            self.noise_schedule
        )
        self.policy.set_noise_scale(noise_scale)

    def step_explore(self, o, **kwargs):
        o = self._get_step_feature(o)
        with self.policy.deterministic_(False):
            a, _ = self.policy.action(o, use_noise_clip=False, **kwargs)
        a = ptu.get_numpy(a)
        return a, {}

    def step_exploit(self, o, **kwargs):
        o = self._get_step_feature(o)
        with self.policy.deterministic_(True):
            a, _ = self.policy.action(o, **kwargs)
        a = ptu.get_numpy(a)
        return a, {}

    def compute_q_target(self, actor_next_obs, next_obs, rewards, terminals, v_pi_kwargs={}):
        with torch.no_grad():
            next_action, _ = self.policy.action(
                actor_next_obs, **self.next_sample_kwargs)
            target_q_next_action = self.qf_target.value(
                next_obs, next_action, **v_pi_kwargs)[0]
            q_target = rewards + (1.-terminals)*self.discount*target_q_next_action
        return q_target
    
    def compute_policy_loss(self, obs, new_action, action_info, v_pi_kwargs={}, prefix='policy/'):
        q_new_action, _ = self.qf.value(
            obs, 
            new_action, 
            return_ensemble=False, 
            **v_pi_kwargs
        )
        q_pi_mean = q_new_action.mean()
        policy_loss = -q_pi_mean
        policy_info = self._log_policy_info(
            new_action, policy_loss, q_pi_mean, prefix)
        return policy_loss, policy_info

    def _log_policy_info(self, new_action, policy_loss, q_pi_mean, prefix):
        policy_info = OrderedDict()
        policy_info[prefix+'loss'] = policy_loss.item()
        policy_info[prefix+'q_pi'] = q_pi_mean.item()
        if self._log_tb_or_not(): 
            for i in range(new_action.shape[-1]):
                logger.tb_add_histogram(
                    'action_%d'%i, 
                    new_action[:,i], 
                    self.num_train_steps
                )
            logger.tb_flush()
        return policy_info

    def train_from_torch_batch(self, batch, only_update_latent=False, train_latent_ratio=None):
        if train_latent_ratio is None:
            train_latent_ratio = self.train_latent_ratio
        shift = self.n_step_td
        frame_stack = self.frame_stack
        assert batch['frames'].shape[1] == frame_stack + shift
        for k in ['rewards', 'terminals', 'actions']:
            assert batch[k].shape[1] == frame_stack+shift-1, k
        batch_size = len(batch['frames'])
        latent_n = int(train_latent_ratio*batch_size)
        f = batch['frames']
        a = batch['actions']
        r = batch['rewards']
        t = batch['terminals']
        f_aug = self.aug_trans(f)
        ################ set noise ################
        self.set_noise_scale()
        self.train_info['noise_scale'] = self.policy.noise_scale
        ######################
        # process raw inpute #
        ######################
        f_1 = f[:latent_n]
        a_1 = a[:latent_n]
        r_1 = r[:latent_n]
        t_1 = t[:latent_n]
        e_1 = self.obs_processor(f_aug[:latent_n])
        if self.multi_step_train:
            z1, z2, p_mean, p_std = self.traj_processor.posterior_process(e_1, a_1)
            q_mean, q_std = self.traj_processor.prior_process(a_1)
        else:
            z1, z2, p_mean, p_std, q_mean, q_std = self.traj_processor.process(e_1, a_1)
        #######################
        # update latent model #
        #######################
        latent_loss, latent_info = self.compute_latent_loss(
            z1, z2, p_mean, p_std, q_mean, q_std,
            f_1, a_1, r_1, t_1
        )
        self.latent_optimizer.zero_grad()
        latent_loss.backward()
        self.log_latent_grad_norm()
        self.latent_optimizer.step()
        self.train_info.update(latent_info)
        self.log_frame_for_latent_training(f_1)
        if only_update_latent:
            self.log_train_info()
            self.num_train_steps += 1
            return self.train_info
        ###################################### for ddpg ######################################
        ######################
        # process raw inpute #
        ######################
        if self.independent_training:
            ddpg_n = batch_size - latent_n
        else:
            ddpg_n = batch_size
        # more efficient without computing gradient
        with torch.no_grad():
            f_2 = f[-ddpg_n:]
            a_2 = a[-ddpg_n:]
            r_2 = r[-ddpg_n:]
            t_2 = t[-ddpg_n:]
            _discount = 1
            _live = 1
            _reward = 0
            for i in range(shift,0,-1):
                _reward = _reward + r_2[:,-i]*_discount
                _live = _live * (1-t_2[:,-i])
                _discount = _discount * self._true_discount * _live
            ddpg_terminal = 1-_live
            ddpg_reward = _reward
            self.log_batch(ddpg_reward, ddpg_terminal)
            e_2 = self.obs_processor(f_aug[-ddpg_n:])
            z1, z2, p_mean, p_std = self.traj_processor.posterior_process(e_2, a_2)
            z = torch.cat([z1, z2], -1)
            obs_actor = torch.cat( 
                [e_2[:,:frame_stack].reshape(ddpg_n, -1), a_2[:,:frame_stack-1].reshape(ddpg_n, -1)], 
                dim=-1)
            next_obs_actor = torch.cat(
                [e_2[:,-frame_stack:].reshape(ddpg_n, -1), a_2[:,-(frame_stack-1):].reshape(ddpg_n, -1)], 
                dim=-1)
        #################
        # update critic #
        #################
        q_target = self.compute_q_target(
            next_obs_actor, 
            z[:,-1], 
            ddpg_reward, 
            ddpg_terminal, 
            self.next_v_pi_kwargs
        )
        qf_loss, train_qf_info = self.compute_qf_loss(z[:,frame_stack-1], a_2[:,frame_stack-1], q_target)
        self.log_frame_for_critic_training(f_2)
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
                obs_actor, **self.current_sample_kwargs)
            # update policy
            policy_loss, train_policy_info = self.compute_policy_loss(
                z[:,frame_stack-1], 
                new_action, 
                action_info, 
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
        if logger.log_or_not(logger.DEBUG):
            self.anomaly_detection()
        batch_keys = list(batch.keys())
        for k in batch_keys:
            del batch[k]
        self.log_train_info()
        self.num_train_steps += 1
        return copy.deepcopy(self.train_info)

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.qf_target,
            self.obs_processor,
            self.traj_processor
        ]

    def get_snapshot(self):
        if self.use_jit_script: # torch.save can not save jit.script_module
            return dict(
                policy=self.policy,
                qf=self.qf,
                qf_target=self.qf_target
            )
        else:
            return dict(
                policy=self.policy,
                qf=self.qf,
                qf_target=self.qf_target,
                obs_processor=self.obs_processor,
                traj_processor=self.traj_processor
            )
        
