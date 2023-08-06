from mirl.components import Component
from mirl.agents.mbpo.mbpo_collector import MBPOCollector
import mirl.torch_modules.utils as ptu
import torch
from mirl.utils.logger import logger

class MOPOCollector(MBPOCollector):
    def __init__(
        self, 
        model, 
        policy, 
        pool,
        imagined_data_pool, 
        penalty_coef=1,
        penalty_type="total_var", # max_var
        **base_kwargs
    ):
        self.penalty_coef = penalty_coef
        super().__init__(
            model=model,
            policy=policy,
            pool=pool,
            imagined_data_pool=imagined_data_pool,
            **base_kwargs
        )
        self.step_kwargs['return_distribution'] = True
        self.penalty_type = penalty_type

    def add_sample(
        self, stop, 
        o, a, next_o, r, d, 
        info, depth, cur_epoch
    ):
        samples = {}
        ind = self._get_live_ind(stop)
        samples['observations'] = o[ind]
        samples['actions'] = a[ind]
        samples['next_observations'] = next_o[ind]
        # samples['rewards'] = r[ind]
        samples['terminals'] = d[ind]
        ####### for mopo #########
        means = info['ensemble_next_obs_mean']
        stds = info['ensemble_next_obs_std']
        with torch.no_grad():
            if self.penalty_type == "max_var":
                penalty = torch.norm(stds,dim=-1,keepdim=True)
                penalty = torch.max(penalty,dim=0)[0]
            elif self.penalty_type == "total_var":
                mean_var = torch.mean(stds**2,dim=0)
                var_mean = torch.var(means,dim=0)
                total_var = mean_var + var_mean
                penalty = torch.sum(total_var,dim=-1,keepdim=True)
        samples['rewards'] = r[ind] - self.penalty_coef*penalty[ind]
        if logger.log_or_not(logger.WARNING):
            prefix = "imagined_samples/depth_%d"%depth
            logger.tb_add_scalar(prefix+"/reward", torch.mean(r[ind]), cur_epoch)
            logger.tb_add_histogram(prefix+"/reward_hist", r[ind], cur_epoch)
            logger.tb_add_scalar(prefix+"/penalty", torch.mean(penalty[ind]), cur_epoch)
            logger.tb_add_histogram(prefix+"/penalty_hist", penalty[ind], cur_epoch)
        ########## end ###########
        for k in samples:
            samples[k] = ptu.get_numpy(samples[k])
        self.imagined_data_pool.add_samples(samples)
        if self.rollout_terminal_state:
            return next_o, d
        else:
            ind = self._get_live_ind(d)
            return next_o[ind], d[ind]



