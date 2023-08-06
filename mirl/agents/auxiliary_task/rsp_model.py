from mirl.processors.base_processor import Processor
from mirl.torch_modules.mlp import MLP
import mirl.torch_modules.utils as ptu
from mirl.utils.logger import logger
from mirl.agents.ddpg_agent import plot_scatter
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch

def generate_coef(gamma, scale=1, seq_len=128, mode="fourier"):
    if mode == "fourier":
        fourier_gamma = np.arange(seq_len)/seq_len*2*np.pi
        real = np.cos(fourier_gamma)
        image = np.sin(-fourier_gamma)
        fourier_gamma = np.stack([real, image])
        gamma = (fourier_gamma*gamma)
        reward_lambda = np.ones(seq_len)*scale
    elif mode == "laplace":
        laplace_gamma = np.arange(seq_len)/seq_len
        laplace_gamma = np.exp(-laplace_gamma)
        gamma = laplace_gamma*gamma
        reward_lambda = np.ones(seq_len)*scale
    elif mode == "direct":
        return 1, gamma
    elif mode == "random":
        rand_gamma = np.random.normal(seq_len, seq_len)
        rand_gamma = ptu.from_numpy(rand_gamma)
        rand_gamma = torch.softmax(rand_gamma,dim=0)
        rand_scale = ptu.rand(1, seq_len)*gamma
        gamma = rand_gamma * rand_scale
        reward_lambda = np.random.rand(1,seq_len)
        reward_lambda = ptu.from_numpy(reward_lambda)
        return reward_lambda, gamma
    else:
        raise NotImplementedError
    gamma = ptu.from_numpy(gamma)
    reward_lambda = ptu.from_numpy(reward_lambda)
    return reward_lambda, gamma

def td_style(reward, next_v, reward_lambda, gamma, mode="fourier"):
    if mode == "fourier":
        k = gamma.shape[-1]
        next_v = next_v.view(-1,2,k)
        real_part = next_v * gamma
        real_part = real_part[:,0]-real_part[:,1]
        real_part = real_part + reward_lambda*reward
        gamma = gamma[[1,0]]
        image_part = next_v * gamma
        image_part = image_part.sum(dim=1)
        v = torch.cat([real_part, image_part], dim=-1)
    elif mode == "laplace":
        v = gamma*next_v+reward_lambda*reward
    elif mode == "direct":
        v = torch.cat([reward, next_v[:,:-1]*gamma],dim=-1)
    elif mode == "random":
        v = torch.mm(reward, reward_lambda) + torch.mm(next_v, gamma)
    else:
        raise NotImplementedError
    return v

class RSPModel(nn.Module, Processor):
    def __init__(
        self,
        env,
        processor,
        embedding_size=50,
        seq_len=128, 
        reward_scale=1, 
        gamma=0.99,
        seq_embed_mode="fourier",
        predictor_layers=[256,256],
        r_coef=1,
        q_coef=1,
        use_target_network=False,
        soft_target_tau=1e-2,
        activation='relu',
    ):
        nn.Module.__init__(self)
        self.action_size = env.action_shape[0]
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        self.seq_embed_mode = seq_embed_mode
        self.pred_seq_len = seq_len*2 if seq_embed_mode=="fourier" else seq_len
        def build_networks():
            trunk = nn.Sequential(
                nn.Linear(processor.output_shape[0], self.embedding_size),
                nn.LayerNorm(self.embedding_size),
                nn.Tanh())
            seq_predictor = MLP(
                embedding_size+self.action_size,
                1+self.pred_seq_len, #r, q
                hidden_layers=predictor_layers,
                activation=activation)
            return trunk, seq_predictor
        self.trunk, self.seq_predictor = build_networks()
        self.use_target_network = use_target_network
        if use_target_network:
            target_networks = build_networks()
            self.target_trunk = target_networks[0]
            self.target_seq_predictor = target_networks[1]
            self._update_target(1)
        else:
            self.target_trunk = self.trunk
            self.target_seq_predictor = self.seq_predictor
        self.soft_target_tau = soft_target_tau
        self.reward_lambda, self.gamma = generate_coef(
            gamma, reward_scale, seq_len, seq_embed_mode)
        self.q_coef = q_coef
        self.r_coef = r_coef
    
    def _update_target(self, tau):
        if not self.use_target_network:
            return
        if tau == 1:
            ptu.copy_model_params_from_to(self.trunk, self.target_trunk)
            ptu.copy_model_params_from_to(self.seq_predictor, self.target_seq_predictor)
        else:
            ptu.soft_update_from_to(self.trunk, self.target_trunk, tau)
            ptu.soft_update_from_to(self.seq_predictor, self.target_seq_predictor, tau)

    def _compute_predict_loss(self, r, tar_r, q, tar_q):
        r_loss = F.mse_loss(r, tar_r)
        q_loss = F.mse_loss(q, tar_q)
        loss = self.r_coef*r_loss + self.q_coef*q_loss
        return loss, r_loss, q_loss

    def compute_auxiliary_loss(self, obs, a, r ,next_obs, 
            next_a, n_step=0, log=False, frame=None):
        self._update_target(self.soft_target_tau)
        #predict 
        h = self.trunk(obs)
        pred_feature = torch.cat([h,a],dim=-1)
        pred_rq = self.seq_predictor(pred_feature)
        pred_r = pred_rq[:,:1]
        pred_q = pred_rq[:,1:]
        with torch.no_grad():
            next_h = self.target_trunk(next_obs)
            next_pred_feature = torch.cat([next_h,next_a],dim=-1)
            next_pred_rq = self.target_seq_predictor(next_pred_feature)
            next_target_q = next_pred_rq[:,1:]
            target_q = td_style(r, next_target_q, self.reward_lambda, 
                                self.gamma, self.seq_embed_mode)
        pred_loss, r_loss, q_loss =self._compute_predict_loss( 
                    pred_r, r.detach(), pred_q, target_q.detach())
        if log:
            plot_scatter(r, pred_r, "x=y", "aux/reward", n_step)
            logger.tb_add_scalar("aux/r_loss", r_loss, n_step)
            logger.tb_add_scalar("aux/q_loss", q_loss, n_step)
        return pred_loss

    def process(self):
        raise NotImplementedError