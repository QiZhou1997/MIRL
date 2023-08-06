import torch
import torch.nn as nn
import torch.nn.functional as F
import mirl.torch_modules.utils as ptu
from mirl.processors.base_processor import TrajectorProcessor
from mirl.processors.cnn_processor import CNNDecoder
from mirl.torch_modules.gaussian_v2 import *
from mirl.policies.gaussian_policy import GaussianPolicy
from mirl.policies.truncnorm_policy import TruncNormPolicy
from torch import nn
import numpy as np

def slac_weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
@torch.jit.script
def calculate_kl_divergence(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) /  q_std).pow_(2)
    kl = 0.5 * (var_ratio + t1 - 1 - var_ratio.log() ) 
    return kl

    

@torch.jit.script
def normal_log_prob(pred, mean, std):
    noise = (pred - mean) / std
    log_p = -0.5*noise.pow(2) - std.log() - 0.5*math.log(2 * math.pi)
    return log_p

class SLACv2Policy(TruncNormPolicy):
    def __init__( 
        self, 
        env, 
        processor,
        deterministic=False,
        policy_name='slac_policy',
        **truncnorm_kwargs
    ):
        self.frame_stack = env.frame_stack
        self._frame_shape = env.frame_shape
        self._channel = self._frame_shape[0]
        self.feature_size = processor.output_shape[0]*self.frame_stack + \
            env.action_space.shape[0]*(self.frame_stack-1) 
        super().__init__(env, deterministic, policy_name=policy_name, **truncnorm_kwargs)

    def _get_feature_size(self):
        return self.feature_size

class SLACPolicy(GaussianPolicy):
    def __init__( 
        self, 
        env, 
        processor,
        deterministic=False,
        squashed=True,
        policy_name='slac_policy',
        **gaussian_kwargs
    ):
        self.frame_stack = env.frame_stack
        self._frame_shape = env.frame_shape
        self._channel = self._frame_shape[0]
        self.feature_size = processor.output_shape[0]*self.frame_stack + \
            env.action_space.shape[0]*(self.frame_stack-1) 
        super().__init__(env, deterministic, squashed, policy_name, **gaussian_kwargs)

    def _get_feature_size(self):
        return self.feature_size

# based on slac_pytorch
class SLACLatentModel(TrajectorProcessor, nn.Module):
    def __init__(
        self,
        env,
        obs_processor,
        z1_dim=32,
        z2_dim=256,
        decode_std=np.sqrt(0.1),
        decoder_kwargs={},
        mlp_kwargs={
            "hidden_layers": [256, 256],
            "init_bias_constant":0,
            "init_func_name": "xavier_uniform_",
            "activation":"leaky_relu_0.2"
        }
    ):
        nn.Module.__init__(self)
        self.action_shape = action_shape = env.action_space.shape
        self.embedding_shape = embedding_shape = obs_processor.output_shape
        obs_processor.apply(slac_weight_init)
        self.frame_shape = env.frame_shape

        self.output_shape = (z1_dim+z2_dim,)
        self.decode_std = decode_std

        # p(z1(0)) = N(0, I)
        self.z1_prior_init = FixedGaussian(z1_dim, 1.0)
        # p(z2(0) | z1(0))
        self.z2_prior_init = MeanSoftplusStdGaussian(
            z1_dim, 
            z2_dim, 
            **mlp_kwargs
        )
        # p(z1(t+1) | z2(t), a(t))
        self.z1_prior = MeanSoftplusStdGaussian(
            z2_dim + action_shape[0],
            z1_dim,
            **mlp_kwargs,
        )
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_prior = MeanSoftplusStdGaussian(
            z1_dim + z2_dim + action_shape[0],
            z2_dim,
            **mlp_kwargs,
        )

        # q(z1(0) | feat(0))
        self.z1_posterior_init = MeanSoftplusStdGaussian(
            embedding_shape[0], 
            z1_dim, 
            **mlp_kwargs
        )
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.z1_posterior = MeanSoftplusStdGaussian(
            embedding_shape[0] + z2_dim + action_shape[0],
            z1_dim,
            **mlp_kwargs
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_posterior = self.z2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward = MeanSoftplusStdGaussian(
            2 * z1_dim + 2 * z2_dim + action_shape[0],
            1,
            **mlp_kwargs,
        )

        # feat(t) = Encoder(x(t))
        self.decoder = CNNDecoder(
            env=env,
            input_size=z1_dim+z2_dim, 
            **decoder_kwargs
        )
        self.apply(slac_weight_init)
        self.can_imagine = False

    def bind(self, actor, value):
        self.actor = actor
        self.value = value
        self.can_imagine = True

    # for jit compile without the actor and value
    def bind_virtual(self): 
        virtual_actor = nn.Linear(self.output_shape[0], self.action_shape[0])
        virtual_value = nn.Linear(self.output_shape[0], 1)
        self.actor = virtual_actor
        self.value = virtual_value

    @torch.jit.export
    def posterior_init(self, e):
        z1, z1_mean, z1_std = self.z1_posterior_init(e)
        z2, _, _ = self.z2_posterior_init(z1)
        return z1, z2, z1_mean, z1_std

    @torch.jit.export
    def prior_process(
        self, 
        actions
    ):
        z1, _, _ = self.z1_prior_init(actions[:, 0])
        return self.conditioned_prior_process(z1, actions)

    @torch.jit.export
    def conditioned_prior_process(
        self, 
        z1,
        actions,
    ):
        z1_mean_ = []
        z1_std_ = []
        # p(z1(0)) = N(0, I)
        _, z1_mean, z1_std = self.z1_prior_init(actions[:, 0])
        # p(z2(0) | z1(0))
        z2, _, _ = self.z2_prior_init(z1)
        ####### 
        z1_mean_.append(z1_mean)
        z1_std_.append(z1_std)
        for t in range(1, actions.size(1) + 1):
            z1, z2, z1_mean, z1_std = self.prior_step(z1, z2, actions[:, t-1])
            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)
        return z1_mean_, z1_std_

    

    @torch.jit.export
    def posterior_process(
        self, 
        embeddings, 
        actions
    ):
        z1_ = []
        z2_ = []
        z1_mean_ = []
        z1_std_ = []
        z1, z2, z1_mean, z1_std = self.posterior_init(embeddings[:, 0])
        z1_mean_.append(z1_mean)
        z1_std_.append(z1_std)
        z1_.append(z1)
        z2_.append(z2)
        for t in range(1, embeddings.shape[1]):
            z1, z2, z1_mean, z1_std = self.posterior_step(
                z1, z2, actions[:, t-1], embeddings[:, t])
            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2) 
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)
        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)
        return z1_, z2_, z1_mean_, z1_std_

    @torch.jit.export
    def process(
        self, 
        embeddings, 
        actions
    ):
        z1_ = []
        z2_ = []
        posterior_mean_ = []
        posterior_std_ = []
        prior_mean_ = []
        prior_std_ = []
        z1, z2, z1_mean, z1_std = self.posterior_init(embeddings[:, 0])
        z1_.append(z1)
        z2_.append(z2)
        posterior_mean_.append(z1_mean)
        posterior_std_.append(z1_std)
        z1_mean, z1_std = self.z1_prior_init.get_mean_std(actions[:, 0])
        prior_mean_.append(z1_mean)
        prior_std_.append(z1_std)
        for t in range(1, embeddings.shape[1]):
            z1, z2, z1_mean, z1_std = self.posterior_step(
                z1, z2, actions[:, t-1], embeddings[:, t])
            z1_.append(z1)
            z2_.append(z2) 
            posterior_mean_.append(z1_mean)
            posterior_std_.append(z1_std)
            # p(z1(t) | z2(t-1), a(t-1))
            z1_feature = torch.cat([z2, actions[:, t-1]], dim=1)
            z1_mean, z1_std = self.z1_prior.get_mean_std(z1_feature)
            prior_mean_.append(z1_mean)
            prior_std_.append(z1_std)
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)
        posterior_mean_ = torch.stack(posterior_mean_, dim=1)
        posterior_std_ = torch.stack(posterior_std_, dim=1)
        prior_mean_ = torch.stack(prior_mean_, dim=1)
        prior_std_ = torch.stack(prior_std_, dim=1)
        return z1_, z2_, posterior_mean_, posterior_std_, prior_mean_, prior_std_

    @torch.jit.export
    def imagine(
        self,
        prev_z1,
        prev_z2,
        horizon: int
    ):
        if not self.can_imagine:
            raise RuntimeError
        # 注意batch和length的位置
        z1, z2 = prev_z1, prev_z2
        feature_ = []
        a_ = []
        feature = torch.cat([z1, z2], dim=-1)
        feature_.append(feature)
        for _ in range(horizon):
            a = self.actor(feature)
            a_.append(a)
            # (state+action) + belief -> new_belief, state
            z1, z2, _, _ = self.prior_step(z1, z2, a)
            feature = torch.cat([z1, z2], dim=-1)
            feature_.append(feature)
        feature_ = torch.stack(feature_, dim=1)
        a_ = torch.stack(a_, dim=1)
        reward_, _ = self.predict_reward(feature_[:,:-1], a_, feature_[:,1:])
        feature_ = feature_.view(-1, self.output_shape[0])
        value_ = self.value(feature_)
        value_ = value_.view(-1, horizon+1, 1)
        return reward_, value_

    @torch.jit.export
    def posterior_step(
        self,
        prev_z1, #not use. maintain a unified interface with RSSM
        prev_z2, 
        prev_action,
        embedding
    ):
        z1_feature = torch.cat([prev_z2, prev_action, embedding], dim=-1)
        z1, z1_mean, z1_std = self.z1_posterior(z1_feature)
        z2_feature = torch.cat([z1, prev_z2, prev_action], dim=1)
        z2, _, _ = self.z2_posterior(z2_feature)
        return z1, z2, z1_mean, z1_std

    @torch.jit.export
    def prior_step(
        self,
        prev_z1,
        prev_z2, 
        prev_action
    ):
        z1_feature = torch.cat([prev_z2, prev_action], dim=1)
        z1, z1_mean, z1_std = self.z1_prior(z1_feature)
        z2_feature = torch.cat([z1, prev_z2, prev_action], dim=1)
        z2, _, _ = self.z2_prior(z2_feature)
        return z1, z2, z1_mean, z1_std 

    @torch.jit.export
    def decode(self, latent):
        mean = self.decoder(latent)
        std = torch.ones_like(mean) * self.decode_std
        return mean, std

    @torch.jit.export
    def predict_reward(self, z, a, next_z):
        B, L, _ = z.shape
        feature = torch.cat([z, a, next_z], dim=-1).reshape(B*L,-1)
        r_mean, r_std = self.reward.get_mean_std(feature)
        r_mean = r_mean.view(B, L, 1)
        r_std = r_std.view(B, L, 1)
        return r_mean, r_std

    def forward(self, frames, actions):
        return self.process(frames, actions)
        
    @torch.jit.export
    def compute_latent_loss(
        self, 
        z1, z2,
        posterior_mean, posterior_std,
        prior_mean, prior_std, 
        frame, 
        action,
        reward,
        free_nats: float
    ):
        # Calculate KL divergence loss.
        kl_divergence = calculate_kl_divergence(
            posterior_mean, posterior_std, 
            prior_mean, prior_std)
        loss_kld = kl_divergence.sum(dim=2)

        # Prediction loss of images.
        frame = frame/255.0-0.5
        z = torch.cat([z1, z2], dim=-1)
        frame_mean, frame_std = self.decode(z)
        log_prob = normal_log_prob(frame, frame_mean, frame_std)
        loss_image = - log_prob.sum(dim=[2,3,4])

        # Prediction loss of rewards.
        r_mean, r_std = self.predict_reward(z[:, :-1], action, z[:, 1:])
        log_prob = normal_log_prob(reward, r_mean, r_std)

        # 没乘terminal因为没看懂
        loss_reward = - log_prob.sum(dim=2)
        return (loss_kld, loss_image, loss_reward), (frame, frame_mean)

    