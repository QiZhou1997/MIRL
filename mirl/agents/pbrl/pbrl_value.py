from mirl.values.ensemble_value import EnsembleQValue
from mirl.torch_modules.mlp import MLP
from torch.nn import Module

class EnsembleWithPrior(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = MLP(*args, **kwargs)
        self.prior_net = MLP(*args, **kwargs)
        for param in self.prior_net.parameters():
            param.requires_grad = False
    
    def forward(self, *args, **kwargs):
        x = self.net(*args, **kwargs)
        prior = self.prior_net(*args, **kwargs)
        return x+prior

class PBRLQValue(EnsembleQValue):
    def __init__( 
        self, 
        env, 
        ensemble_size=2,
        use_prior=True,
        value_name='ensemble_q_value',
        **mlp_kwargs
    ):
        super().__init__(env, ensemble_size, value_name=value_name,**mlp_kwargs)
        self.use_prior = use_prior
        if self.use_prior:
            self.module = EnsembleWithPrior( 
                self._get_feature_size(), 
                1,
                ensemble_size=ensemble_size,
                module_name=value_name,
                **mlp_kwargs
            )