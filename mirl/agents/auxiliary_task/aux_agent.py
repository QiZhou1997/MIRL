from collections import OrderedDict

import numpy as np
from numpy.core.fromnumeric import argmin
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

import mirl.torch_modules.utils as ptu
from mirl.agents.ddpg_agent import DDPGAgent
from mirl.agents.drq.drq_agent import DrQAgent, log_frame
from mirl.processors.utils import *
from mirl.utils.logger import logger
from mirl.utils.misc_untils import get_scheduled_value
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import pdb
import copy

#TODO： 用小的batch训练latent是否是因为出于计算效率/显存考虑？
#actor和critic的训练需要较大的batch，而序列梯度回传只能较小的batch？
#请测试前向不计算梯度和后向不计算梯度的耗时和显存开销
class AuxiliaryTaskAgent(DrQAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        processor,
        auxiliary_model,
        aux_coef=1,
        use_offline=True,
        policy_share_trunk=True,
        qf_share_trunk=False,
        detach_qf_obs=True,
        **drq_kwargs
    ):
        super().__init__(env, policy, qf, qf_target,processor,**drq_kwargs)
        if policy_share_trunk:
            policy.trunk = auxiliary_model.trunk
            self.policy_optimizer = self.optimizer_class(
                self.policy.module.parameters(),
                lr=self.policy_lr,
                **self.optimizer_kwargs
            )
        if qf_share_trunk:
            qf.trunk = auxiliary_model.trunk
        self.auxiliary_model = auxiliary_model
        self.aux_coef = aux_coef
        combination_parameters = []
        combination_parameters += qf.parameters()
        combination_parameters += processor.parameters()
        combination_parameters += auxiliary_model.parameters()
        self.combination_optimizer = self.optimizer_class(
            combination_parameters,
            lr=self.qf_lr,
            **self.optimizer_kwargs
        )
        self.combination_lr = self.qf_lr
        self.detach_qf_obs = detach_qf_obs
        self.use_offline = use_offline
    
    def compute_q_target(self, actor_next_obs, next_obs, rewards, terminals, v_pi_kwargs={}):
        with torch.no_grad():
            alpha = self._get_alpha()
            next_action, next_policy_info = self.policy.action(
                actor_next_obs, **self.next_sample_kwargs
            )
            log_prob_next_action = next_policy_info['log_prob']
            target_q_next_action = self.qf_target.value(
                next_obs, 
                next_action, 
                **v_pi_kwargs
            )[0] - alpha * log_prob_next_action
            q_target = rewards + (1.-terminals)*self.discount*target_q_next_action
        return q_target, next_action

    def train_from_torch_batch(self, batch):
        rewards = batch['rewards'][:,-2]
        terminals = batch['terminals'][:,-2]
        cur_frames = batch['frames'][:,:3]
        N, _, C, H, W = cur_frames.shape
        cur_frames = cur_frames.reshape(N,3*C,H,W)
        actions = batch['actions'][:,-2]
        next_frames = batch['frames'][:,1:4]
        N, _, C, H, W = next_frames.shape
        next_frames = next_frames.reshape(N,3*C,H,W)
        offline_next_actions = batch['actions'][:,-1]
        self.log_batch(rewards, terminals)
        ################
        # augmentation #
        ################
        batch_size = len(next_frames)
        frame_aug = []
        next_frame_aug = []
        for _ in range(self.n_aug):
            frame_aug.append(self.aug_trans(cur_frames))
            next_frame_aug.append(self.aug_trans(next_frames))
        frame_aug = torch.cat(frame_aug, dim=0)
        next_frame_aug = torch.cat(next_frame_aug, dim=0)
        rewards_aug = rewards.repeat(self.n_aug,1)
        terminals_aug = terminals.repeat(self.n_aug,1)
        actions_aug = actions.repeat(self.n_aug,1)
        offline_next_actions_aug = offline_next_actions.repeat(self.n_aug,1)
        #process frames
        cat_frame_aug = torch.cat([frame_aug,next_frame_aug])
        cat_obs_aug = self.processor(cat_frame_aug)
        obs_aug, actor_next_obs_aug = torch.chunk(cat_obs_aug,2)
        if self.target_processor is None:
            next_obs_aug = actor_next_obs_aug
        else:
            next_obs_aug = self.target_processor(next_frame_aug)
        #################
        # update critic #
        #################
        # compute target
        q_target, next_actions_aug = self.compute_q_target(
            actor_next_obs_aug,
            next_obs_aug, 
            rewards_aug, 
            terminals_aug, 
            self.next_v_pi_kwargs
        )
        q_target = q_target.reshape((self.n_aug,batch_size,1)).mean(0)
        q_target_aug = q_target.repeat(self.n_aug,1)
        # compute loss
        if self.detach_qf_obs:
            value_obs_aug = obs_aug.detach()
        else:
            value_obs_aug = obs_aug
        qf_loss, train_qf_info = self.compute_qf_loss(
            value_obs_aug, 
            actions_aug, 
            q_target_aug)
        self.train_info.update(train_qf_info)
        self.log_frame(frame_aug, next_frame_aug)
        ################
        # update model #
        ################
        if self.use_offline:
            next_actions_aug = offline_next_actions_aug
        model_loss = self.auxiliary_model.compute_auxiliary_loss(
            obs_aug, actions_aug, rewards_aug, next_obs_aug, next_actions_aug, 
            n_step=self.num_train_steps, 
            log=self._log_tb_or_not(), 
            frame=frame_aug)
        ############## end ##############
        # self.qf_optimizer.zero_grad()
        # qf_loss.backward()
        # self.log_critic_grad_norm()
        # self.qf_optimizer.step()
        self.combination_optimizer.zero_grad()
        (qf_loss+self.aux_coef*model_loss).backward()
        self.log_critic_grad_norm()
        self.combination_optimizer.step()
        if self.num_train_steps % self.target_update_freq == 0:
            self._update_target(self.soft_target_tau)
        ###########################
        # update actor and alpha #
        ###########################
        if self.num_train_steps % self.policy_update_freq == 0:
            actor_obs = obs_aug.detach()
            new_action, action_info = self.policy.action(
                actor_obs, **self.current_sample_kwargs
            )
            # update alpha
            if self.use_automatic_entropy_tuning:
                alpha_loss, train_alpha_info = self.compute_alpha_loss(action_info)
                self.train_info.update(train_alpha_info)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
            # update policy
            policy_loss, train_policy_info = self.compute_policy_loss(
                actor_obs,  #note
                new_action, 
                action_info, 
                self.current_v_pi_kwargs
            )
            policy_loss = policy_loss + (new_action**2).mean()*0.03
            self.train_info.update(train_policy_info)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
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
        nets = [
            self.policy,
            self.qf,
            self.processor,
            self.auxiliary_model
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf=self.qf,
            obs_processor=self.processor,
            aux_net=self.auxiliary_model
        )