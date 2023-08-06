from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim 
import torch.nn.functional as F
import mirl.torch_modules.utils as ptu
from mirl.agents.sac_agent import SACAgent
from mirl.utils.process import Progress, Silent, format_for_process
from mirl.utils.eval_util import create_stats_ordered_dict
from mirl.utils.logger import logger
import copy

# TODO algorithm get_item替代getattr __dict__
class FisherBRCAgent(SACAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        bc_steps=0,
        bp_lr_milestone=[int(8e5), int(9e5)],
        bp_lr_decay=0.1,
        bp_init_lr=1e-3,
        bp_train_steps=int(1e6),
        bp_train_batch_size=256,
        bp_train_silent=False,
        fisher_reg=0.1,
        live_bonus=5,
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
        self.qf_optimizer = self.optimizer_class(
            self.qf.module.parameters(),
            lr=self.qf_lr,
            **self.optimizer_kwargs
        )
        self.bp_optimizer = self.optimizer_class(
            self.qf.bp.parameters(),
            lr=bp_init_lr,
            **self.optimizer_kwargs
        )
        self.bp_lr_schedule=optim.lr_scheduler.MultiStepLR(
            self.bp_optimizer, 
            bp_lr_milestone, 
            gamma=bp_lr_decay, 
            last_epoch=-1
        )
        self.bc_steps = bc_steps
        self.fisher_reg = fisher_reg
        self.live_bonus = live_bonus
        self.progress_class = Silent if bp_train_silent else Progress
        self.bp_train_steps = bp_train_steps
        self.bp_train_batch_size = bp_train_batch_size
        self.log_bp_alpha = ptu.FloatTensor([self.log_alpha.item()])
        self.log_bp_alpha.requires_grad_(True)
        self.bp_alpha_optimizer = self.optimizer_class(
            [self.log_bp_alpha],
            lr=bp_init_lr,
            **self.optimizer_kwargs
        )

    def pretrain(self, pool):
        progress = self.progress_class(self.bp_train_steps)
        for i in range(self.bp_train_steps):
            progress.update()
            torch_batch = pool.random_batch_torch(self.bp_train_batch_size)
            params = self.train_bp_from_torch_batch(torch_batch, i)
            self.bp_lr_schedule.step()
            progress.set_description(format_for_process(params))
        self._update_target(1)
        self.qf.fix_bp()
        self.qf_target.fix_bp()

    def train_bp_from_torch_batch(self, batch, nth_pretrain):
        obs = batch['observations']
        actions = batch['actions']
        _,bp_info = self.qf.bp(obs,return_log_prob=True)
        log_prob_bp_action = bp_info['log_prob']
        bp_entropy = -log_prob_bp_action.mean()
        bp_alpha = self.log_bp_alpha.exp().detach()
        train_bp_log_prob = self.qf.log_prob(obs, actions)
        train_bp_log_prob = train_bp_log_prob.mean()
        bp_loss = -train_bp_log_prob-bp_alpha*bp_entropy
        self.bp_optimizer.zero_grad()
        bp_loss.backward()
        self.bp_optimizer.step()
        # alpha
        res = (self.target_entropy-bp_entropy).detach()
        bp_alpha_loss = (-res) * self.log_bp_alpha.exp()
        self.bp_alpha_optimizer.zero_grad()
        bp_alpha_loss.backward()
        self.bp_alpha_optimizer.step()
        # log
        bp_lr = self.bp_lr_schedule.get_lr()[0]
        results = {
            'pre/bp_log_p': train_bp_log_prob.item(), 
            'pre/bp_loss': bp_loss.item(), 
            "pre/bp_alpha": self.log_bp_alpha.exp().item(),
            "pre/res": res.item(),
            "pre/bp_alpha_loss": bp_alpha_loss.item(),
            "pre/bp_entropy": bp_entropy.item(),
            "pre/bp_lr": bp_lr
        }
        if nth_pretrain % self.tb_log_freq == 0:
            for k,v in results.items():
                logger.tb_add_scalar(k, v, nth_pretrain)
        return results

    def compute_q_target(self, next_obs, rewards, terminals, v_pi_kwargs={}):
        with torch.no_grad():
            next_action, _ = self.policy.action(next_obs, **self.next_sample_kwargs)
            target_q_next_action = self.qf_target.value(
                next_obs, 
                next_action, 
                **v_pi_kwargs
            )[0]
            q_target = rewards + (1.-terminals)*self.discount*target_q_next_action
        return q_target

    def train_from_torch_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        rewards = rewards + self.live_bonus
        self.log_batch(rewards, terminals)
        #################
        # update critic #
        #################
        q_target = self.compute_q_target(next_obs, rewards, terminals, self.next_v_pi_kwargs)
        qf_loss, train_qf_info = self.compute_qf_loss(obs, actions, q_target)
        ############# for FisherBRC ###############
        with torch.no_grad():
            all_obs = torch.cat([obs,next_obs])
            # actions
            all_action,_ = self.policy.action(
                all_obs, 
                reparameterize=False,
                return_log_prob=False,
            )
        all_action.requires_grad_()
        _, all_value_info = self.qf.value(
            all_obs, 
            all_action,
            return_offset_gradient=True, 
            return_logprob=True,
            only_return_ensemble=True
        )
        grad_list = all_value_info['offset_grad_list']
        grad_reg = 0
        for grad in grad_list:
            grad = (grad**2).sum(-1)
            grad_reg = grad_reg+grad.mean()
        grad_reg = grad_reg * self.fisher_reg
        bp_log_prob = all_value_info['bp_log_prob']
        with torch.no_grad():
            if logger.log_or_not(logger.INFO):
                self.train_info.update(create_stats_ordered_dict(
                    'fbrc/bp_log_prob',
                    ptu.get_numpy(bp_log_prob),
                ))
                self.train_info.update(create_stats_ordered_dict(
                    'fbrc/grad',
                    ptu.get_numpy(grad),
                ))
        ############# FisherBRC end ###############
        self.qf_optimizer.zero_grad()
        (qf_loss+grad_reg).backward()
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
