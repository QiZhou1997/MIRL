from mirl.policies.gaussian_policy import GaussianPolicy
import mirl.torch_modules.utils as ptu
from torch import nn

class LatentGaussianPolicy(GaussianPolicy):
    def __init__( 
        self, 
        env, 
        processor,
        deterministic: bool = False,
        squashed: bool = True,
        policy_name: str = 'latent_policy',
        **mlp_kwargs
    ):
        self.feature_size = processor.output_shape[0]
        super().__init__(env, deterministic, squashed, policy_name, **mlp_kwargs)
 
    def _get_feature_size(self):
        return self.feature_size

class TEPolicy(GaussianPolicy):
    def __init__( 
        self, 
        env, 
        obs_processor,
        traj_processor,
        deterministic=False,
        squashed=True,
        policy_name='te_q_value',
        **mlp_kwargs
    ):
        self.frame_stack = env.frame_stack
        self._frame_shape = env.frame_shape
        self._channel = self._frame_shape[0]
        self.feature_size = traj_processor.output_shape[0]
        super().__init__(env, deterministic, squashed, policy_name, **mlp_kwargs)
        self.obs_processor = obs_processor
        self.traj_processor = traj_processor



if __name__ == "__main__":
    pass

