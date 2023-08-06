from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim 
import torch.nn.functional as F
import mirl.torch_modules.utils as ptu
from mirl.agents.cql_agent import CQLAgent
from mirl.utils.misc_untils import combine_item
from mirl.utils.logger import logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import copy

# TODO algorithm get_item替代getattr __dict__
class COMPOAgent(CQLAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        penalize_only_imagined=False,
        **cql_kwargs
    ):
        super().__init__(
            env,
            policy,
            qf,
            qf_target,
            **cql_kwargs
        )
        self.penalize_only_imagined = penalize_only_imagined

    def train_from_torch_batch(self, real_batch, imagined_batch):
        batch = combine_item(real_batch, imagined_batch)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        self.log_batch(rewards, terminals)
        batch_size = obs.shape[0]
        act_size = actions.shape[1]
        real_batch_size = len(real_batch)
        #################
        # update critic #
        #################
        q_target = self.compute_q_target(next_obs, rewards, terminals, self.next_v_pi_kwargs)
        qf_loss, q_pred, train_qf_info = self.compute_qf_loss(obs, actions, q_target)
        ############# for CQL ###############
        with torch.no_grad():
            # observations
            if self.penalize_only_imagined:
                cql_obs = imagined_batch['observations']
                cql_next_obs = imagined_batch['next_observations']
            else:
                cql_obs = obs
                cql_next_obs = next_obs
            cql_batch_size = len(cql_obs)
            repeat_obs = cql_obs.repeat(self.sample_number, 1)
            repeat_next_obs = cql_next_obs.repeat(self.sample_number, 1)
            repeat_cat_obs = torch.cat([repeat_obs,repeat_next_obs])
            all_obs = cql_obs.repeat(self.sample_number*3, 1)
            # actions
            repeat_cat_action,info = self.policy.action(
                repeat_cat_obs, 
                reparameterize=False,
                return_log_prob=True,
            )
            random_actions = ptu.rand(cql_batch_size*self.sample_number,act_size)*2-1
            all_actions = torch.cat([repeat_cat_action, random_actions])
            # logprob for inportance sampling
            logprob = info['log_prob']
            cont = np.log(0.5**act_size)
            random_logprob = ptu.ones(cql_batch_size*self.sample_number,1)*cont
            logprob = torch.cat([logprob, random_logprob])
        _, all_value_info = self.qf.value(
            all_obs, 
            all_actions.detach(),
            return_ensemble=True, 
            only_return_ensemble=True
        )
        reg_q = all_value_info['ensemble_value']
        min_reg = (reg_q-logprob).view(-1, self.sample_number*3, cql_batch_size, 1)
        min_reg = torch.logsumexp(min_reg/self.cql_temp, dim=1)
        min_reg = min_reg.mean()*self.cql_temp
        cql_reg = min_reg - q_pred[:,:real_batch_size].mean()
        cql_reg = cql_reg * self.cql_reg_coef
        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            cql_reg = alpha_prime*cql_reg
            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = alpha_prime*(self.target_action_gap-cql_reg)
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        with torch.no_grad():
            rep_num = batch_size*self.sample_number
            self.train_info['cql/cql_reg'] = cql_reg.item()
            self.train_info['cql/random_value'] = reg_q[:,-rep_num:].mean().item()
            self.train_info['cql/cur_value'] = reg_q[:,:rep_num].mean().item()
            self.train_info['cql/next_value'] = reg_q[:,rep_num:rep_num*2].mean().item()
            self.train_info['cql/q_pred'] = q_pred.mean().item()
            if self.with_lagrange:
                self.train_info['cql/alpha_value'] = alpha_prime.item()
                self.train_info['cql/alpha_loss'] = alpha_prime_loss.item()
        ############# CQL end ###############
        self.qf_optimizer.zero_grad()
        (qf_loss+cql_reg).backward()
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
