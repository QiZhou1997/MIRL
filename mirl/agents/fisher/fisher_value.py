import torch 
from mirl.torch_modules.gaussian import *
from mirl.torch_modules.mlp import MLP
from mirl.values.ensemble_value import EnsembleQValue, sample_from_ensemble
import numpy as np

class MixedGaussian(MLP):
    def __init__(
        self, 
        feature_size: int , 
        out_pred_size: int , 
        component_number: int = 5,
        bound_mode: str = 'tanh',
        logstd_extra_bias: float = 0,
        init_func_name: str = "orthogonal_",
        module_name: str = 'mean_logstd_gaussian',
        squashed: bool = True,
        mean_scale: float = 1.0,
        std_scale: float = 1.0,
        min_std: float = 1e-4,
        **mlp_kwargs
    ) -> None:
        super().__init__(
            feature_size,
            out_pred_size*2,
            ensemble_size=component_number,
            module_name=module_name,
            init_func_name=init_func_name,
            **mlp_kwargs
        )
        self.feature_size = feature_size
        self.out_pred_size = out_pred_size
        self.bound_mode = bound_mode
        self.logstd_extra_bias = logstd_extra_bias
        self.squashed = squashed
        self.mean_scale = mean_scale
        self.std_scale = std_scale
        self.cn = component_number
        assert min_std >= 0
        self.min_std = min_std

    def get_mean_std(self, x):
        output=MLP.forward(self, x)
        mean, log_std = torch.chunk(output,2,-1)
        log_std = log_std + self.logstd_extra_bias
        log_std = bound_logstd(log_std, self.bound_mode)
        return mean, torch.exp(log_std)

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        reparameterize: bool = True,
        return_log_prob: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        mean, std = self.get_mean_std(x)
        mean = mean * self.mean_scale
        std = std*self.std_scale + self.min_std
        dist = Normal(mean, std)
        if deterministic:
            raise NotImplementedError
        elif reparameterize:
            out_pred = dist.rsample()
        else:
            out_pred = dist.sample()
        if self.squashed:
            pretanh = out_pred
            out_pred = torch.tanh(out_pred)
        batch_size = len(x)
        ind = np.random.randint(self.cn, size=(batch_size,))
        arange = np.arange(batch_size)
        out_pred = out_pred[ind, arange]
        info = {}
        if return_log_prob:
            if self.squashed:
                _pretanh = pretanh[ind, arange].expand(self.cn,-1,-1)
                log_prob = dist.log_prob(_pretanh)
                #对称会更beautiful
                log_prob -= 2*np.log(2)-F.softplus(2*_pretanh)-F.softplus(-2*_pretanh)
            else:
                _out_pred = out_pred.expand(self.cn,-1,-1)
                log_prob = dist.log_prob(_out_pred)
            log_prob = log_prob.sum(-1, keepdim=True)
            log_prob = torch.logsumexp(log_prob, dim=0)-np.log(self.cn)
            info['log_prob'] = log_prob
        return out_pred, info

    def log_prob(
        self, 
        x: torch.Tensor, 
        out_pred: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        mean, std = self.get_mean_std(x)
        dist = Normal(mean, std)
        if self.squashed:
            pretanh = atanh(out_pred, eps)
            pretanh = pretanh.expand(self.cn,-1,-1)
            log_prob = dist.log_prob(pretanh)
            log_prob = log_prob - torch.log(1-out_pred**2+eps)
        else:
            log_prob = dist.log_prob(out_pred)
        log_prob = log_prob.sum(-1, keepdim=True)
        log_prob = torch.logsumexp(log_prob, dim=0)-np.log(self.cn)
        return log_prob

class FisherValue(EnsembleQValue):
    def __init__( 
        self, 
        env, 
        ensemble_size=2,
        value_name='ensemble_q_value',
        behavior_policy_kwargs={},
        **mlp_kwargs
    ):
        super().__init__(
            env, 
            ensemble_size=ensemble_size,
            value_name=value_name,
            **mlp_kwargs
        )
        self.bp = MixedGaussian(
            self.observation_shape[0],
            self.action_shape[0],
            **behavior_policy_kwargs
        )

    def log_prob(self, s, a):
        return self.bp.log_prob(s,a)

    def fix_bp(self):
        for param in self.bp.parameters():
            param.requires_grad = False

    def value(
        self, 
        obs, 
        action, 
        sample_number=2, 
        batchwise_sample=False,
        mode='min', 
        return_ensemble=False,
        only_return_ensemble=False,
        return_offset_gradient=False,
        return_logprob=False
    ):
        input_tensor = self._get_features(obs, action)
        if self.ensemble_size is not None:
            ensemble_offset = self.module(input_tensor)
            bp_logprob = self.bp.log_prob(obs, action)
            ensemble_value = ensemble_offset+bp_logprob
            assert ensemble_value.shape[-1] == 1
            info = {}
            if return_offset_gradient:
                grad_list = []
                for i in range(self.ensemble_size):
                    offset = ensemble_offset[i]
                    grad = torch.autograd.grad(
                        outputs=offset, 
                        inputs=action,
                        grad_outputs=torch.ones_like(offset),
                        create_graph=True, 
                        retain_graph=True,
                        only_inputs=True
                    )[0]
                    assert grad.shape[-1] == action.shape[-1]
                    assert grad.shape[0] == action.shape[0]
                    grad_list.append(grad)
                info['offset_grad_list'] = grad_list
            if return_logprob:
                info['bp_log_prob'] = bp_logprob
                assert bp_logprob.shape[-1] == 1
                assert bp_logprob.shape[0] == action.shape[0]
            if return_ensemble and self.ensemble_size is not None:
                info['ensemble_value'] = ensemble_value
            if only_return_ensemble:
                return None, info
            if sample_number is None:
                sample_number = self.ensemble_size
            if self.ensemble_size != sample_number:
                if batchwise_sample:
                    index = np.random.choice(self.ensemble_size, sample_number, replace=False)
                    sampled_value = ensemble_value[...,index,:,:]
                else:
                    sampled_value = sample_from_ensemble(ensemble_value, sample_number, replace=False)
            else:
                sampled_value = ensemble_value
            if mode == 'min':
                value = torch.min(sampled_value, dim=-3)[0]
            elif mode == 'mean':
                value = torch.mean(sampled_value, dim=-3)
            elif mode == 'max':
                value = torch.max(sampled_value, dim=-3)[0]
            elif mode == 'sample':
                index = np.random.randint(sample_number)
                value = sampled_value[index]
            else:
                raise NotImplementedError
        else:
            value = self.module(input_tensor)
        return value, info