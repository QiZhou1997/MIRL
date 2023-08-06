import torch
from torch import nn
from torch.nn import functional as F
from mirl.agents.mbpo.base_model import Model
from mirl.torch_modules.mlp import MLP
from mirl.utils.logger import logger
import mirl.torch_modules.utils as ptu

import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG2PI = np.log(2*np.pi)


def gaussian_log_prob(mean, log_std, target):
    log_var = log_std*2
    inv_var = torch.exp(-log_var)
    diff = target - mean
    mse = diff ** 2
    log_prob = -0.5 * (LOG2PI + log_var + mse*inv_var)
    return log_prob, diff

class PEModelModule(MLP):
    def __init__(
        self,
        obs_size, 
        action_size, 
        learn_reward=True,
        learn_done=False,
        ensemble_size=None,
        model_name='PE', 
        reward_coef=1,
        bound_reg_coef=2e-2,
        weight_decay=[2.5e-5,5e-5,7.5e-5,1e-5,7.5e-5],
        # weight_decay=0,
        **mlp_kwargs
    ):
        assert not learn_done 
        self.obs_size = obs_size
        self.action_size = action_size
        self.learn_reward = learn_reward
        self.learn_done = learn_done
        if self.learn_reward:
            output_size = (obs_size+1)*2 #mean_obs, mean_r, std_obs, std_r, 
            self.mean_std_size = obs_size+1
        else:
            output_size = obs_size*2 #mean_obs, std_obs
            self.mean_std_size = obs_size
        self.obs_size = obs_size
        self.action_size = action_size
        super().__init__(
            obs_size+action_size,
            output_size, 
            ensemble_size=ensemble_size,
            module_name=model_name,
            **mlp_kwargs
        )
        self.reward_coef = reward_coef
        self.bound_reg_coef = bound_reg_coef
        self.weight_decay = weight_decay
        self.max_log_std, self.min_log_std = [], []
        for i in range(self.ensemble_size):
            _max_log_std = nn.Parameter(ptu.ones(1, self.mean_std_size, dtype=torch.float32) / 4.0)
            _min_log_std = nn.Parameter(-ptu.ones(1, self.mean_std_size, dtype=torch.float32) * 5.0)
            self.max_log_std.append(_max_log_std)
            self.min_log_std.append(_min_log_std)
            setattr(self, "maxlogstd_net%d"%i, _max_log_std)
            setattr(self, "minlogstd_net%d"%i, _min_log_std)

    def compute_loss(
        self,
        obs,
        action,
        delta,
        reward=None,
        done=None,
        ignore_terminal_state=False,
        n_step=0,
        log=False
    ):
        l2_reg = self.get_weight_decay(self.weight_decay)
        bound_reg = self.bound_reg_coef*self.get_extra_loss()
        s_log_p, r_log_p, s_diff, r_diff = self.log_prob(obs, action, delta, reward, done)
        if ignore_terminal_state:
            assert done is not None
            s_log_p = s_log_p * (1-done)
            if self.learn_reward:
                r_log_p = r_log_p * (1-done)
        s_loss = -s_log_p.mean([1,2])
        if self.learn_reward:
            r_loss = self.reward_coef * -r_log_p.mean([1,2])
        else:
            r_loss = 0
        ensemble_loss = s_loss + r_loss 
        loss = ensemble_loss.sum(0) + l2_reg + bound_reg
        if log:
            if self.learn_reward:
                r_diff = torch.abs(r_diff).mean()
            else:
                r_diff = 0
            logger.tb_add_scalar("mt/r_diff", r_diff, n_step)
            logger.tb_add_scalar("mt/s_diff", torch.abs(s_diff).mean(), n_step)
            logger.tb_add_scalar("mt/r_loss", r_loss.mean(), n_step)
            logger.tb_add_scalar("mt/s_loss", s_loss.mean(), n_step)
            logger.tb_add_scalar("mt/l2_reg", l2_reg, n_step)
            logger.tb_add_scalar("mt/bound_reg", bound_reg, n_step)
        return loss, ensemble_loss
        
    def get_extra_loss(self):
        reg = sum(self.max_log_std) - sum(self.min_log_std)
        reg = torch.sum(reg)
        return reg

    # for normalized delta
    def log_prob(
        self,
        obs,
        action,
        delta,
        reward=None,
        done=None
    ):
        input_tensor = torch.cat([obs, action], dim=-1)
        output = super().forward(input_tensor)
        assert output.dim() == 3
        mean, log_std = self._get_mean_logstd(output)
        s_log_p, s_diff = gaussian_log_prob(
            mean[...,:self.obs_size],
            log_std[...,:self.obs_size],
            delta
        )
        s_log_p = s_log_p.sum(dim=-1, keepdim=True)
        if self.learn_reward:
            assert reward is not None and self.reward_coef>0
            r_log_p, r_diff = gaussian_log_prob(
                mean[...,-1:],
                log_std[...,-1:],
                reward
            )
            return s_log_p, r_log_p, s_diff, r_diff
        else:
            return s_log_p, None, s_diff, None
    
    def _get_mean_logstd(self, tensor):
        mean,log_std = torch.chunk(tensor, 2, dim=-1)
        max_log_std = torch.stack(self.max_log_std)
        min_log_std = torch.stack(self.min_log_std)
        log_std = max_log_std - F.softplus(max_log_std-log_std)
        log_std = min_log_std + F.softplus(log_std-min_log_std)
        return mean, log_std

    def forward(
        self, 
        obs, 
        action,
        elite_indices=None,
        return_distribution=False
    ):
        # NOTE: we predict (normalized) delta (next_obs - obs) instead of next_obs
        assert obs.dim() == action.dim()
        input_tensor = torch.cat([obs, action], dim=-1)
        all_mean_logstd = super().forward(input_tensor)
        assert all_mean_logstd.dim() == 3
        batch_size = all_mean_logstd.shape[1]
        # sample a gaussian dist
        if elite_indices is None:
            elite_indices = list(np.arange(self.ensemble_size))
        indices = np.random.choice(elite_indices, batch_size)
        n_arange = np.arange(batch_size)
        all_mean, all_logstd = self._get_mean_logstd(all_mean_logstd)
        all_std = torch.exp(all_logstd)
        mean = all_mean[indices, n_arange]
        std = all_std[indices, n_arange]
        # sample from gaussian
        deltas_rewards = torch.randn_like(mean)*std + mean      
        deltas = deltas_rewards[...,:self.obs_size]
        if self.learn_reward:
            rewards = deltas_rewards[...,-1:]
        else:
            rewards = None
        info = {}
        if return_distribution:
            info['ensemble_delta_mean'] = all_mean[...,:self.obs_size]
            info['ensemble_delta_std'] = all_std[...,:self.obs_size]
            if self.learn_reward:
                info['ensemble_reward_mean'] = all_mean[...,-1:]
                info['ensemble_reward_std'] = all_std[...,-1:]
        return deltas, rewards, None, info

class PEModel(Model):
    def __init__(
        self,  
        env,
        normalize_obs=True,
        normalize_action=True,
        normalize_delta=True,
        normalize_reward=False,
        known=['done'],
        ensemble_size=7,
        allow_to_use=None,
        elite_number=5,
        model_name='PE_model',
        **module_kwargs
    ):
        Model.__init__(
            self,  
            env,
            normalize_obs,
            normalize_action,
            normalize_delta,
            normalize_reward,
            known
        )
        assert len(self.processed_obs_shape)==1 and len(self.processed_action_shape)==1
        self.ensemble_size = ensemble_size
        self.elite_number = elite_number
        self.module = PEModelModule(
            self.processed_obs_shape[0], 
            self.processed_action_shape[0],
            self.learn_reward,
            self.learn_done,
            self.ensemble_size,
            model_name,
            **module_kwargs
        )
        if allow_to_use is None:
            allow_to_use = self.ensemble_size
        self.allow_to_use = allow_to_use
        self.elite_indices = None
        self.rank = list(np.arange(ensemble_size))
        self.loss = None
    
    def remember_loss(self, loss):
        self.loss = loss
        self.rank = np.argsort(loss[:self.allow_to_use])
        self.elite_indices = list(self.rank[:self.elite_number])

    def _predict_delta(
        self, 
        obs, 
        action,
        return_distribution=False
    ):
        return self.module(
            obs, 
            action,
            self.elite_indices,
            return_distribution
        )

    def step(self, obs, action, **kwargs):
        next_obs, r, d, info = super().step(obs, action, **kwargs)
        #recover output by normalizer
        keys = list(info.keys())
        for k in keys:
            v = info[k]
            if v.dim() == 3:
                v = v[self.elite_indices]
            info[k] = v
            if 'delta' in k:
                new_k = k.replace('delta', 'next_obs')
                if 'std' in k:
                    if self.normalize_delta:
                        std, epsilon = self.delta_processor.std, self.delta_processor.epsilon
                        v = v*(std+epsilon)
                        info[k] = v
                    info[new_k] = v
                else:
                    if self.normalize_delta:
                        v = self.delta_processor.recover(v)
                        info[k] = v
                    if v.dim() == obs.dim()+1:
                        info[new_k] = v + obs[None,:,:]
                    else:
                        info[new_k] = v + obs
            if self.learn_reward and self.normalize_reward and 'reward' in k:
                if 'std' in k:
                    std, epsilon = self.reward_processor.std, self.reward_processor.epsilon
                    info[k] = v*(std+epsilon)
                else:
                    info[k] = self.reward_processor.recover(v)
        return next_obs, r, d, info

    def compute_loss(
        self,
        obs,
        action,
        delta,
        reward=None,
        done=None,
        ignore_terminal_state=True,
        n_step=0,
        log=False
    ):
        if self.normalize_obs:
            obs = self.obs_processor.process(obs)
        if self.normalize_action:
            action = self.action_processor.process(action)
        if self.normalize_delta:
            delta = self.delta_processor.process(delta)
        if self.learn_reward and self.normalize_reward:
            reward = self.reward_processor.process(reward)
        return self.module.compute_loss(
            obs, action, delta, reward, done,
            ignore_terminal_state, n_step, log
        )
        
    def log_prob_np(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, save_dir=None, net_id=None):
        if save_dir == None:
            save_dir = logger._snapshot_dir
        self.module.save(save_dir, net_id)
    
    def load(self, load_dir=None, net_id=None):
        if load_dir == None:
            load_dir = logger._snapshot_dir
        return self.module.load(load_dir, net_id)

    def get_snapshot(self):
        return self.module.get_snapshot()

