from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim 
import torch.nn.functional as F
import mirl.torch_modules.utils as ptu
from mirl.agents.sac_agent import SACAgent
from mirl.utils.misc_untils import get_scheduled_value
from mirl.utils.logger import logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import copy

# TODO algorithm get_item替代getattr __dict__
class PBRLAgent(SACAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        sample_number=10,
        indist_reg_coef=0.01,
        outdist_reg_coef=[4000,50000,5,0.2],
        bc_steps = 100000,
        next_sample_kwargs={
            'reparameterize':False, 
            'return_log_prob':False
        },
        **sac_kwargs
    ):
        super().__init__(
            env,
            policy,
            qf,
            qf_target,
            next_sample_kwargs=next_sample_kwargs,
            **sac_kwargs
        )
        self.sample_number = sample_number
        self.indist_reg_coef = indist_reg_coef
        self.outdist_reg_coef = outdist_reg_coef
        self.bc_steps = bc_steps
        self.next_v_pi_kwargs["return_ensemble"] = True
        self.next_v_pi_kwargs["only_return_ensemble"] = True


    def compute_q_target(self, next_obs, rewards, terminals, v_pi_kwargs={}):
        with torch.no_grad():
            next_action, _ = self.policy.action(next_obs, **self.next_sample_kwargs)
            _,value_info = self.qf_target.value(
                next_obs, 
                next_action, 
                **v_pi_kwargs
            )
            pbrl_nextq = value_info['ensemble_value']
            in_penalty = torch.std(pbrl_nextq, dim=0)
            pbrl_nextq = pbrl_nextq - self._incoef*in_penalty
            pbrl_nextq = torch.clamp_min(pbrl_nextq, 0)
            q_target = rewards + (1.-terminals)*self.discount*pbrl_nextq
        return q_target

    def compute_qf_loss(self, obs, actions, q_target):
        _, value_info = self.qf.value(obs, actions, return_ensemble=True)
        q_value_ensemble = value_info['ensemble_value']
        qf_loss = F.mse_loss(q_value_ensemble, q_target)
        qf_info = self._log_q_info(q_target, q_value_ensemble, qf_loss)
        return qf_loss, q_value_ensemble, qf_info
    
    def set_reg_coef(self):
        self._incoef = get_scheduled_value(
            self.num_train_steps, 
            self.indist_reg_coef
        )
        self._outcoef = get_scheduled_value(
            self.num_train_steps, 
            self.outdist_reg_coef
        )

    def train_from_torch_batch(self, batch):
        self.set_reg_coef()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        self.log_batch(rewards, terminals)
        #################
        # update critic #
        #################
        q_target = self.compute_q_target(next_obs, rewards, terminals, self.next_v_pi_kwargs)
        qf_loss, q_pred, train_qf_info = self.compute_qf_loss(obs, actions, q_target)
        ############# for PBRL ###############
        with torch.no_grad():
            # observations
            repeat_obs = obs.repeat(self.sample_number, 1)
            repeat_next_obs = next_obs.repeat(self.sample_number, 1)
            all_obs = torch.cat([repeat_obs,repeat_next_obs])
            # actions
            all_action,_ = self.policy.action(
                all_obs, 
                reparameterize=False,
                return_log_prob=False,
            )
        _, all_value_info = self.qf.value(
            all_obs, 
            all_action.detach(),
            return_ensemble=True, 
            only_return_ensemble=True
        )
        reg_q = all_value_info['ensemble_value']
        with torch.no_grad():
            out_penalty = torch.std(reg_q,dim=0)
            reg_q_target = reg_q - self._outcoef*out_penalty
            reg_q_target = torch.clamp_min(reg_q_target, 0)
        pbrl_reg = F.mse_loss(reg_q, reg_q_target.detach())
        with torch.no_grad():
            self.train_info['pbrl/pbrl_reg'] = pbrl_reg.item()
            self.train_info['pbrl/out_q'] = reg_q.mean().item()
            self.train_info['pbrl/in_q'] = q_pred.mean().item()
        ############# PBRL end ###############
        self.qf_optimizer.zero_grad()
        (qf_loss+pbrl_reg).backward()
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
            #update alpha
            if self.use_automatic_entropy_tuning:
                alpha_loss, train_alpha_info = self.compute_alpha_loss(action_info)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.train_info.update(train_alpha_info)
            #update policy
            bc_log_prob = self.policy.log_prob(obs,actions).mean()
            self.train_info['policy/bc_log_prob'] = bc_log_prob.item()
            policy_loss, train_policy_info = self.compute_policy_loss(
                obs, 
                new_action, 
                action_info, 
                self.current_v_pi_kwargs
            )
            if self.num_train_steps < self.bc_steps:
                alpha = self._get_alpha()
                new_log_prob = action_info['log_prob'].mean()
                policy_loss = alpha*new_log_prob - bc_log_prob
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
