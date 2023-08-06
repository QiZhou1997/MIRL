from mirl.processors.base_processor import Processor
from mirl.torch_modules.mlp import MLP
import mirl.torch_modules.utils as ptu
from mirl.utils.logger import logger
from mirl.agents.ddpg_agent import plot_scatter
from mirl.utils.misc_untils import get_scheduled_value
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch

def generate_coef(seq_len=128, mode="learned"):
    assert mode in ['learned', 'random']
    lam_scale = ptu.randn(seq_len)
    gamma_transfer = ptu.randn(seq_len, seq_len)
    gamma_scale = ptu.randn(seq_len,)
    if mode == 'learned':
        lam_scale = nn.Parameter(lam_scale)
        # gamma_transfer = nn.Parameter(gamma_transfer)
        gamma_scale = nn.Parameter(gamma_scale)
    return lam_scale, gamma_transfer, gamma_scale

def td_style(reward, next_v, gamma, lam_scale, gamma_transfer, gamma_scale, tmp=1, control=True):
    if control:
        gamma_transfer = torch.softmax(gamma_transfer/tmp, dim=0)
        gamma_scale = gamma*torch.sigmoid(gamma_scale)
    # lam_scale = F.softplus(lam_scale)+1
    target_r = reward*lam_scale
    target_next = torch.mm(next_v, gamma_transfer)*gamma_scale 
    target_q = target_r+target_next
    return target_q, target_r, target_next

def compute_cl_loss(e1, e2, alpha, n_aug=2):
    e1_norm = torch.norm(e1, dim=-1, p=2, keepdim=True) 
    e1 = e1 / e1_norm
    e2_norm = torch.norm(e2, dim=-1, p=2, keepdim=True) 
    e2 = e2 / e2_norm
    similarity = torch.mm(e1, torch.t(e2))
    similarity = similarity/alpha
    with torch.no_grad():
        pred_prob = torch.softmax(similarity, dim=-1)
        target_prob = ptu.eye(len(similarity))
        if n_aug==2:
            aug_p = ptu.eye(len(similarity))
            p1, p2 = torch.chunk(aug_p,2,dim=-1)
            aug_p = torch.cat([p2, p1], dim=-1)
            target_prob = (target_prob+aug_p)/2
        accuracy = (pred_prob * target_prob).sum(-1)
        diff = pred_prob-target_prob
    loss = (similarity*diff).sum(-1).mean()
    return loss, pred_prob, accuracy

class LearnedRSPModel(nn.Module, Processor):
    def __init__(
        self,
        env,
        processor,
        embedding_size=50,
        seq_len=128, 
        reward_scale=1,
        n_aug=2,
        cl_coef=0.0,
        gamma=0.9,
        seq_embed_mode="learned",
        predictor_layers=[256,256],
        r_coef=1,
        q_coef=1,
        use_target_network=False,
        soft_target_tau=1e-2,
        tmp = 1,
        alpha=0.05,
        momentum=0.001,
        activation='relu',
        control=True,
    ):
        nn.Module.__init__(self)
        self.action_size = env.action_shape[0]
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        self.seq_embed_mode = seq_embed_mode
        self.pred_seq_len = seq_len
        self.control = control
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
        self.n_aug = n_aug
        self.gamma = gamma
        self.tmp = tmp
        self.alpha = alpha
        self.cl_coef = cl_coef
        self.lam_scale, self.gamma_transfer, self.gamma_scale = generate_coef(
            seq_len, seq_embed_mode)
        self.q_coef = q_coef
        self.r_coef = r_coef
        self.momentum = momentum
        self._lam_scale, self._gamma_transfer, self._gamma_scale = None, None, None
    
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
        if self._lam_scale is None:
            self._lam_scale = self.lam_scale.detach()
            self._gamma_transfer = self.gamma_transfer.detach()
            self._gamma_scale = self.gamma_scale.detach()
        else:
            mm = self.momentum
            self._lam_scale = self.lam_scale.detach()*mm+(1-mm)*self._lam_scale
            self._gamma_transfer = self.gamma_transfer.detach()*mm+(1-mm)*self._gamma_transfer
            self._gamma_scale = self.gamma_scale.detach()*mm+(1-mm)*self._gamma_scale
        # target_q, target_r, target_next = td_style(r, next_target_q, self.gamma, 
        #         self._lam_scale, self._gamma_transfer, self._gamma_scale, self.tmp)
        target_q, target_r, target_next = td_style(r, next_target_q, self.gamma, 
            self.lam_scale, self.gamma_transfer, self.gamma_scale, self.tmp, self.control)
        target_q[...,-1] = 1
        # end
        pred_loss, r_loss, q_loss =self._compute_predict_loss( 
                    pred_r, r.detach(), pred_q, target_q.detach())
        # _, _, q_loss =self._compute_predict_loss( 
        #             pred_r, r.detach(), pred_q.detach(), target_q)
        if self.cl_coef > 0 and self.seq_embed_mode=='learned':
            cl_loss, _, accuracy = compute_cl_loss(pred_q, 
                                    target_q, self.alpha, self.n_aug)
            pred_loss = pred_loss + cl_loss*self.cl_coef
        if log:
            plot_scatter(r, pred_r, "x=y", "aux/reward", n_step)
            logger.tb_add_scalar("aux/r_loss", r_loss, n_step)
            logger.tb_add_scalar("aux/q_loss", q_loss, n_step)
            logger.tb_add_histogram("aux/lam_scale", self.lam_scale, n_step)
            logger.tb_add_histogram("aux/gamma_scale", self.gamma_scale, n_step)
            logger.tb_add_histogram("aux/target_q", target_q.mean(dim=0), n_step)
            logger.tb_add_histogram("aux/target_r", target_r.mean(dim=0), n_step)
            logger.tb_add_histogram("aux/target_next", target_next.mean(dim=0), n_step)
            if self.cl_coef > 0 and self.seq_embed_mode=='learned':
                logger.tb_add_scalar("aux/accuracy", accuracy.mean(), n_step)
        return pred_loss

    def process(self):
        raise NotImplementedError