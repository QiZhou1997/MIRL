from collections import OrderedDict
import numpy as np
from sklearn import ensemble
import torch
import torch.optim as optim 
import torch.nn.functional as F
import mirl.torch_modules.utils as ptu
from mirl.agents.sac_agent import SACAgent
from mirl.utils.eval_util import create_stats_ordered_dict
from mirl.utils.misc_untils import get_scheduled_value
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
        ood_bar=[-10,10],
        bc_steps=40000,
        qunc_coef=1,
        runc_coef=[4000,50000,5,0.2],
        punish_next_obs=True,
        use_min=True,
        **sac_kwargs
    ):
        super().__init__(
            env,
            policy,
            qf,
            qf_target,
            **sac_kwargs
        )
        self.model = model
        self.bc_steps = bc_steps
        self.ood_bar = ood_bar
        self.qunc_coef = qunc_coef
        self.runc_coef = runc_coef
        self.punish_next_obs = punish_next_obs
        self.current_v_pi_kwargs['use_min']=use_min

    def set_reg_coef(self):
        self._qcoef = get_scheduled_value(
            self.num_train_steps, 
            self.qunc_coef
        )
        self.next_v_pi_kwargs['unc_penalty'] = self._qcoef
        self.current_v_pi_kwargs['unc_penalty'] = self._qcoef
        self._rcoef = get_scheduled_value(
            self.num_train_steps, 
            self.runc_coef
        )

    def compute_q_target(self, next_obs, rewards, terminals, v_pi_kwargs={}):
        next_action, _ = self.policy.action(next_obs, **self.next_sample_kwargs)
        next_q = self.qf_target.value(
            next_obs, 
            next_action, 
            **v_pi_kwargs
        )[0]
        q_target = rewards + (1.-terminals)*self.discount*next_q
        return q_target, next_action

    def compute_qf_loss(
        self, 
        q_value_ensemble, 
        q_target, 
        reward_penalty=0,
        clamp_bar=None, 
        pre="qf/"
    ):
        q_target_expand = q_target.detach().expand(q_value_ensemble.shape)
        with torch.no_grad():
            if reward_penalty != 0:
                # r_penalty = q_value_ensemble.std(dim=-1, keepdim=True)
                r_penalty = q_target_expand.std(dim=-1, keepdim=True)
                q_target_expand = q_target_expand - r_penalty*reward_penalty
            if clamp_bar is not None:
                diff = q_target_expand-q_value_ensemble
                diff = torch.clamp(diff, *clamp_bar)
                q_target_expand = (diff + q_value_ensemble).detach()
        qf_loss = F.mse_loss(q_value_ensemble, q_target_expand)
        qf_info = self._log_q_info(q_target_expand, q_value_ensemble, qf_loss, pre)
        return qf_loss, qf_info
    
    def _log_q_info(self, q_target, q_value_ensemble, qf_loss, pre):
        """
        LOG FOR Q FUNCTION
        """
        qf_info = OrderedDict()
        qf_info[pre+'loss'] = np.mean(ptu.get_numpy(qf_loss))
        if logger.log_or_not(logger.WARNING):
            q_pred = torch.mean(q_value_ensemble, dim=0)
            q_pred_mean = q_pred.mean(dim=-1)
            q_pred_std = q_pred.std(dim=-1)
            qf_info.update(create_stats_ordered_dict(
                pre+'pred_mean',
                ptu.get_numpy(q_pred_mean),
            ))
            qf_info.update(create_stats_ordered_dict(
                pre+'pred_std',
                ptu.get_numpy(q_pred_std),
            ))
        if self._log_tb_or_not():
            q_value_ensemble = q_value_ensemble.mean(0).transpose(0,1)[:3]
            q_target = q_target.mean(0).transpose(0,1)[:3]
            diff = q_value_ensemble - q_target
            info = {}
            info['q_value_ensemble'] = q_value_ensemble
            info['q_target'] = q_target
            info['diff'] = diff
            info['qf_loss'] = qf_loss
            self.plot_value_scatter(info, prefix=pre)
            logger.tb_add_histogram('diff', diff[0], self.num_train_steps)
            logger.tb_add_histogram('q_value_ensemble', q_value_ensemble[0], self.num_train_steps)
            logger.tb_add_histogram('q_target', q_target, self.num_train_steps)
            logger.tb_flush()
        return qf_info

    def train_from_torch_batch(self, real_batch, imagined_batch):
        self.set_reg_coef()
        self.train_info['omega/runc_coef'] = self._rcoef
        self.train_info['omega/qunc_coef'] = self._qcoef
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
            _, _, _, info = self.model.step(o, a, return_distribution=True)
            no_m = info['ensemble_next_obs_mean']
            no_s = info['ensemble_next_obs_std']
            r_m = info['ensemble_reward_mean']
            r_s = info['ensemble_reward_std']
            no = torch.randn_like(no_s)*no_s + no_m
            r = torch.randn_like(r_s)*r_s + r_m
            t = self.model.done_f(None, None, no)
            # reshape
            n_model = r.shape[0]
            batch_size = r.shape[1]
            no = no.view(n_model*batch_size,-1)
            r = r.view(n_model*batch_size,-1)
            t = t.view(n_model*batch_size,-1)
            obs = torch.cat([o, ro])
            actions = torch.cat([a, ra])
            next_obs = torch.cat([no, rno])
            rewards = torch.cat([r, rr])
            terminals = torch.cat([t, rt])
            q_target, next_actions = self.compute_q_target(
                next_obs, rewards, terminals, self.next_v_pi_kwargs)
            ood_q_target = q_target[:n_model*batch_size].view(n_model,batch_size)
            ood_q_target = torch.transpose(ood_q_target, 0, 1)
            id_q_target = q_target[n_model*batch_size:]
        with torch.no_grad():
            target_std = torch.std(ood_q_target,dim=-1)
            if self._log_tb_or_not:
                logger.tb_add_histogram('omega/Tstd_hist', target_std, self.num_train_steps)
            self.train_info.update(create_stats_ordered_dict(
                'omega/Tstd',
                ptu.get_numpy(target_std),
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
        cur_size = len(obs)
        if self.punish_next_obs:
            cat_obs = torch.cat([obs, next_obs])
            cat_actions = torch.cat([actions, next_actions])
        else:
            cat_obs = obs
            cat_actions = actions
        _, value_info = self.qf.value(cat_obs, cat_actions, only_return_ensemble=True)
        cat_ensemble_value = value_info['ensemble_value']
        ensemble_value = cat_ensemble_value[:,:cur_size]
        if self.punish_next_obs:
            next_ensemble_value = cat_ensemble_value[:,cur_size:]
            with torch.no_grad():
                next_ensemble_value = next_ensemble_value.transpose(0,1)
                shape = (next_ensemble_value.shape[0],-1)
                next_ensemble_value = next_ensemble_value.reshape(shape)
                next_uncertainty = torch.std(next_ensemble_value, dim=-1, keepdim=True)
                next_target = next_ensemble_value - next_uncertainty*self._qcoef
            reg = F.mse_loss(next_ensemble_value, next_target)
            self.train_info['omega/reg'] = reg.item()
        else:
            reg = 0
        oodqf_loss, train_oodqf_info = self.compute_qf_loss(
            ensemble_value[:,:-rbatch_size],
            ood_q_target,
            self._rcoef,
            self.ood_bar,
            pre='oodqf/'
        )
        idqf_loss, train_idqf_info = self.compute_qf_loss(
            ensemble_value[:,-rbatch_size:],
            id_q_target,
            0,
            None,
            pre='indf/'
        )
        qf_loss = oodqf_loss + idqf_loss + reg
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
            policy_loss, train_policy_info = self.compute_policy_loss(
                o, 
                new_action, 
                action_info, 
                self.current_v_pi_kwargs
            )
            if self.num_train_steps < self.bc_steps:
                bc_log_prob = self.policy.log_prob(ro,ra).mean()
                alpha = self._get_alpha()
                new_log_prob = action_info['log_prob'][:rbatch_size].mean()
                policy_loss = alpha*new_log_prob - bc_log_prob
            else:
                with torch.no_grad():
                    bc_log_prob = self.policy.log_prob(ro,ra).mean()
            self.train_info['policy/bc_log_prob'] = bc_log_prob.item()
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
