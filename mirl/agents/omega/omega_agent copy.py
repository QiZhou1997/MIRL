from collections import OrderedDict
import numpy as np
from sklearn import ensemble
import torch
import torch.optim as optim 
import torch.nn.functional as F
import mirl.torch_modules.utils as ptu
from mirl.agents.sac_agent import SACAgent
from mirl.utils.misc_untils import combine_item
from mirl.utils.eval_util import create_stats_ordered_dict
from mirl.utils.logger import logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import copy

# TODO algorithm get_item替代getattr __dict__
class OmegaAgent(SACAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        model,
        model_lr=0,
        penalize_type="std",
        penalty_coef=1,
        ood_bar=[-10,10],
        id_bar=[-100,100],
        bc_steps=40000,
        eval_model_number=2,
        **sac_kwargs
    ):
        super().__init__(
            env,
            policy,
            qf,
            qf_target,
            **sac_kwargs
        )
        self.model=model
        self.model_lr = model_lr
        self.model_optimizer = self.optimizer_class(
            self.policy.parameters(),
            lr=model_lr,
            **self.optimizer_kwargs
        )
        self.penalize_type = penalize_type
        self.penalty_coef = penalty_coef
        self.bc_steps = bc_steps
        self.ood_bar = ood_bar
        self.id_bar = id_bar
        self.eval_model_number = eval_model_number

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

    def compute_eval_q_target(self, next_obs, rewards, terminals, v_pi_kwargs={}):
        with torch.no_grad():
            next_action, _ = self.policy.action(next_obs, **self.next_sample_kwargs)
            target_q_next_action = self.qf_target.evaluate(
                next_obs, 
                next_action, 
                **v_pi_kwargs
            )
            q_target = rewards + (1.-terminals)*self.discount*target_q_next_action
        return q_target

    def compute_qf_loss(self, q_value_ensemble, q_target, clamp_bar=None, pre="qf/"):
        q_target_expand = q_target.detach().expand(q_value_ensemble.shape)
        if clamp_bar is not None:
            with torch.no_grad():
                diff = q_target_expand-q_value_ensemble
                diff = torch.clamp(diff, *clamp_bar)
                q_target_expand = (diff + q_value_ensemble).detach()
        qf_loss = F.mse_loss(q_value_ensemble, q_target_expand)
        qf_info = self._log_q_info(q_target, q_value_ensemble, qf_loss, pre)
        return qf_loss, qf_info
    
    def _log_q_info(self, q_target, q_value_ensemble, qf_loss, pre):
        """
        LOG FOR Q FUNCTION
        """
        qf_info = OrderedDict()
        qf_info[pre+'loss'] = np.mean(ptu.get_numpy(qf_loss))
        if logger.log_or_not(logger.WARNING):
            qf_info.update(create_stats_ordered_dict(
                    pre+'target',
                    ptu.get_numpy(q_target),
                ))
            q_pred_mean = torch.mean(q_value_ensemble, dim=0)
            qf_info.update(create_stats_ordered_dict(
                pre+'pred_mean',
                ptu.get_numpy(q_pred_mean),
            ))
            if q_value_ensemble.shape[0] > 1:
                q_pred_std = torch.std(q_value_ensemble, dim=0)
                qf_info.update(create_stats_ordered_dict(
                    pre+'pred_std',
                    ptu.get_numpy(q_pred_std),
                ))
        if self._log_tb_or_not():
            diff = q_value_ensemble - q_target
            info = {}
            info['q_value_ensemble'] = q_value_ensemble
            info['q_target'] = q_target
            info['diff'] = diff
            info['qf_loss'] = qf_loss
            self.plot_value_scatter(info, prefix=pre)
            logger.tb_add_histogram(pre+'diff', diff[0], self.num_train_steps)
            logger.tb_add_histogram(pre+'q_value_ensemble', q_value_ensemble[0], self.num_train_steps)
            logger.tb_add_histogram(pre+'q_target', q_target, self.num_train_steps)
            logger.tb_flush()
        return qf_info

    def train_from_torch_batch(self, real_batch, imagined_batch):
        n_eval = self.eval_model_number
        assert self.model_lr == 0
        ro = real_batch['observations']
        ra = real_batch['actions']
        rr = real_batch['rewards']
        rt = real_batch['terminals']
        rno = real_batch['next_observations']
        io = imagined_batch['observations']
        if len(io) == 0:
            o = ro
        else:
            o = torch.cat([ro, io])
        rbatch_size = len(ro)
        # newest prediction
        with torch.no_grad():
            a, _ = self.policy.action(o)
        if self.model_lr == 0:
            with torch.no_grad():
                _, _, _, info = self.model.step(o, a, return_distribution=True)
                no_m = info['ensemble_next_obs_mean']
                no_s = info['ensemble_next_obs_std']
                r_m = info['ensemble_reward_mean']
                r_s = info['ensemble_reward_std']
                no = torch.randn_like(no_s)*no_s + no_m
                r = torch.randn_like(r_s)*r_s + r_m
                t = self.model.done_f(None, None, no)
                if n_eval > 0:
                    eval_no = no[-n_eval:]
                    eval_r = r[-n_eval:]
                    eval_t = t[-n_eval:]
                    no = no[:-n_eval]
                    r = r[:-n_eval]
                    t = t[:-n_eval]
                # reshape
                n_model = r.shape[0]
                batch_size = r.shape[1]
                no = no.view(n_model*batch_size,-1)
                r = r.view(n_model*batch_size,-1)
                t = t.view(n_model*batch_size,-1)
                next_obs = torch.cat([no, rno])
                rewards = torch.cat([r, rr])
                terminals = torch.cat([t, rt])
                q_target = self.compute_q_target(next_obs, rewards, terminals, self.next_v_pi_kwargs)
                all_ood_q_target = q_target[:n_model*batch_size].view(n_model,batch_size,1)
                id_q_target = q_target[n_model*batch_size:]
                if self.penalize_type == "min":
                    ood_q_target = torch.min(all_ood_q_target, dim=0)[0]
                    ood_q_target = torch.clamp_min(ood_q_target, 0)
                elif self.penalize_type == "std":
                    qtar_mean = torch.mean(all_ood_q_target, dim=0)
                    qtar_std = torch.std(all_ood_q_target, dim=0)
                    ood_q_target = qtar_mean-qtar_std*self.penalty_coef
                    ood_q_target = torch.clamp_min(ood_q_target, 0)
                if n_eval > 0:
                    eval_no = eval_no.view(n_eval*batch_size,-1)
                    eval_r = eval_r.view(n_eval*batch_size,-1)
                    eval_t = eval_t.view(n_eval*batch_size,-1)
                    eval_next_obs = torch.cat([eval_no, rno])
                    eval_rewards = torch.cat([eval_r, rr])
                    eval_terminals = torch.cat([eval_t, rt])
                    eval_q_target = self.compute_eval_q_target(
                        eval_next_obs, 
                        eval_rewards,
                        eval_terminals, 
                        self.next_v_pi_kwargs
                    )
                    eval_all_ood_q_target = eval_q_target[:n_eval*batch_size]
                    eval_all_ood_q_target = eval_all_ood_q_target.view(n_eval,batch_size,1)
                    eval_ood_q_target = torch.mean(eval_all_ood_q_target, dim=0)
                    eval_id_q_target = eval_q_target[n_eval*batch_size:]

        with torch.no_grad():
            std = torch.std(all_ood_q_target,dim=0)
            if self._log_tb_or_not:
                logger.tb_add_histogram('omega/std_hist', std, self.num_train_steps)
            self.train_info.update(create_stats_ordered_dict(
                'omega/std',
                ptu.get_numpy(std),
            ))
            id_mean = id_q_target.mean()
            aood_mean = ood_q_target[:rbatch_size].mean() 
            self.train_info['omega/id_qtar'] = id_mean.item()
            self.train_info['omega/aood_qtar'] = aood_mean.item()
            self.train_info['omega/aood_sub_id'] = (aood_mean-id_mean).item()
            if len(io) > 0:
                sood_mean = ood_q_target[rbatch_size:].mean()
                self.train_info['omega/sood_qtar'] = sood_mean.item()
                self.train_info['omega/sood_sub_id'] = (sood_mean-id_mean).item()
            if self._log_tb_or_not():
                eval_std = torch.std(eval_all_ood_q_target,dim=0)
                if self._log_tb_or_not:
                    logger.tb_add_histogram('omega/eval_std_hist', eval_std, self.num_train_steps)
                self.train_info.update(create_stats_ordered_dict(
                    'omega/eval_std',
                    ptu.get_numpy(eval_std),
                ))
                eval_id_mean = eval_id_q_target.mean()
                eval_aood_mean = eval_ood_q_target[:rbatch_size].mean() 
                eval_info = {}
                eval_info['omega/Eid_qtar'] = eval_id_mean.item()
                eval_info['omega/Eaood_qtar'] = eval_aood_mean.item()
                eval_info['omega/Eaood_sub_id'] = (eval_aood_mean-eval_id_mean).item()
                if len(io) > 0:
                    eval_sood_mean = eval_ood_q_target[rbatch_size:].mean()
                    eval_info['omega/Esood_qtar'] = eval_sood_mean.item()
                    eval_info['omega/Esood_sub_id'] = (eval_sood_mean-eval_id_mean).item()
                for k in eval_info:
                    logger.tb_add_scalar(k, eval_info[k], self.num_train_steps)
        obs = torch.cat([o, ro])
        actions = torch.cat([a, ra])
        _, value_info = self.qf.value(obs, actions, return_ensemble=True)
        ensemble_value = value_info['ensemble_value']
        oodqf_loss, train_oodqf_info = self.compute_qf_loss(
            ensemble_value[:,:-rbatch_size],
            ood_q_target,
            self.ood_bar,
            pre='oodqf/'
        )
        idqf_loss, train_idqf_info = self.compute_qf_loss(
            ensemble_value[:,-rbatch_size:],
            id_q_target,
            self.id_bar,
            pre='indf/'
        )
        if n_eval > 0:
            eval_value = self.qf.evaluate(obs, actions)
            eval_value = eval_value[None]
            eval_ood_loss, eval_ood_info = self.compute_qf_loss(
                eval_value[:,:-rbatch_size],
                eval_ood_q_target,
                pre='Eoodqf/'
            )
            eval_id_loss, eval_id_info = self.compute_qf_loss(
                eval_value[:,-rbatch_size:],
                eval_id_q_target,
                pre='Eindf/'
            )
        else:
            eval_ood_loss = eval_id_loss = 0
            eval_ood_info = eval_id_info = {}
        qf_loss = oodqf_loss + idqf_loss + eval_ood_loss + eval_id_loss
        #################
        # update critic #
        #################
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.log_critic_grad_norm()
        self.qf_optimizer.step()
        if self.num_train_steps % self.target_update_freq == 0:
            self._update_target(self.soft_target_tau)
        self.train_info.update(train_oodqf_info)
        self.train_info.update(train_idqf_info)
        self.train_info.update(eval_ood_info)
        self.train_info.update(eval_id_info)
        ###########################
        # update actor and alpha #
        ###########################
        if self.num_train_steps % self.policy_update_freq == 0:
            new_action, action_info = self.policy.action(
                o, **self.current_sample_kwargs
            )
            #update alpha
            if self.use_automatic_entropy_tuning:
                alpha_loss, train_alpha_info = self.compute_alpha_loss(action_info)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.train_info.update(train_alpha_info)
            #update policy
            bc_log_prob = self.policy.log_prob(ro,ra).mean()
            self.train_info['policy/bc_log_prob'] = bc_log_prob.item()
            policy_loss, train_policy_info = self.compute_policy_loss(
                o, 
                new_action, 
                action_info, 
                self.current_v_pi_kwargs
            )
            if self.num_train_steps < self.bc_steps:
                alpha = self._get_alpha()
                new_log_prob = action_info['log_prob'][:rbatch_size].mean()
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
