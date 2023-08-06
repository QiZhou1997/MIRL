from torch.nn.modules.linear import Linear
from mirl.processors.base_processor import Processor
from mirl.torch_modules.mlp import MLP
from mirl.torch_modules.cnn import CNNTrans
from mirl.utils.logger import logger
from mirl.agents.ddpg_agent import plot_scatter
from mirl.agents.drq.drq_agent import log_frame
import mirl.torch_modules.utils as ptu
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch


class DeepMDPModel(nn.Module, Processor):
    def __init__(
        self,
        env,
        processor,
        embedding_size=50,
        activation='relu',
    ):
        nn.Module.__init__(self)
        self.action_size = env.action_shape[0]
        self.embedding_size = embedding_size
        self.trunk = nn.Sequential(
            nn.Linear(processor.output_shape[0], self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.Tanh())
        self.projector = MLP(
            self.embedding_size+self.action_size,
            self.embedding_size+1, 
            hidden_layers=[256,256],
            activation=activation)

    def _compute_predict_loss(self, r, tar_r, q, tar_q):
        r_loss = F.mse_loss(r, tar_r)
        h_loss = F.mse_loss(q, tar_q)
        loss = r_loss + h_loss
        return loss, r_loss, h_loss

    def compute_auxiliary_loss(self, obs, a, r ,next_obs, 
            next_a, n_step=0, log=False, frame=None):
        #predict 
        h = self.trunk(obs)
        pred_feature = torch.cat([h,a],dim=-1)
        pred_rq = self.seq_predictor(pred_feature)
        pred_r = pred_rq[:,:1]
        pred_h = pred_rq[:,1:]
        target_h = self.trunk(next_obs)
        pred_loss, r_loss, h_loss =self._compute_predict_loss( 
                    pred_r, r.detach(), pred_h, target_h.detach())
        if log:
            plot_scatter(r, pred_r, "x=y", "aux/reward", n_step)
            logger.tb_add_scalar("aux/r_loss", r_loss, n_step)
            logger.tb_add_scalar("aux/h_loss", h_loss, n_step)
        return pred_loss

    def process(self):
        raise NotImplementedError

def get_cnn_trans_kernels(image_size, cnn_kernels):
    last_size = image_size
    cnn_trans_kernels = []
    default = [None, None, None, 1, 0]
    for i, ck in enumerate(cnn_kernels):
        assert len(ck) <= 5 and len(ck) >= 3
        ck = ck + default[len(ck):]
        pad = (last_size+2*ck[4]-ck[2]) % ck[3]
        last_size = (last_size+2*ck[4]-ck[2]) // ck[3] +1
        tck = [j for j in ck]
        tck[0] = ck[1]
        tck[1] = ck[0]
        tck.append(pad)
        cnn_trans_kernels.insert(0, tck)
    return cnn_trans_kernels

#note: normalize target
def build_decoder(
    processor,
    activation='relu'
):
    frame_shape = processor.input_shape
    assert frame_shape[1] == frame_shape[2]
    latent_shape = processor.cnn_net.latent_shape
    cnn_kernels = processor.cnn_net.cnn_kernels
    cnn_trans_kernels = get_cnn_trans_kernels(frame_shape[1], cnn_kernels)
    decoder = CNNTrans(
        latent_shape, 
        frame_shape,
        cnn_trans_kernels=cnn_trans_kernels, 
        activation=activation
    ).to(ptu.device)
    return decoder


class VAEModel(nn.Module, Processor):
    def __init__(
        self,
        env,
        processor,
        embedding_size=50,
        beta=1e-7,
        activation='relu',
    ):
        nn.Module.__init__(self)
        self.action_size = env.action_shape[0]
        self.embedding_size = embedding_size
        self.trunk = nn.Sequential(
            nn.Linear(processor.output_shape[0], self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.Tanh())
        self.decoder1 = nn.Sequential(
            nn.Linear(self.embedding_size, processor.output_shape[0]),
            ptu.get_activation(activation)
        )
        self.decoder2 = build_decoder(processor, activation)
        self.beta = beta

    def decode(self, h):
        h = self.decoder1(h)
        f = self.decoder2.process(h)
        return f

    def compute_auxiliary_loss(self, obs, a, r ,next_obs, 
            next_a, n_step=0, log=False, frame=None):
        #predict 
        h = self.trunk(obs)
        pred_frame = self.decode(h)
        frame = frame/255.0 - 0.5
        recon_loss = F.mse_loss(pred_frame, frame.detach())
        loss = recon_loss + self.beta*(h*h).sum(-1).mean()
        if log:
            log_frame(frame, 0, "orgin", n_step, prefix="vae/")
            log_frame(pred_frame, 0, "recon", n_step, prefix="vae/")
            logger.tb_add_scalar("aux/loss", loss, n_step)
        return loss

    def process(self):
        raise NotImplementedError


def compute_cosine_similarity(e1, e2):
    e1_norm = torch.norm(e1, dim=-1, p=2, keepdim=True) 
    e1 = e1 / e1_norm
    e2_norm = torch.norm(e2, dim=-1, p=2, keepdim=True) 
    e2 = e2 / e2_norm
    similarity = torch.mm(e1, torch.t(e2))
    return similarity

def compute_cl_loss(e1, e2, alpha):
    similarity = compute_cosine_similarity(e1, e2)
    similarity = similarity/alpha
    with torch.no_grad():
        pred_prob = torch.softmax(similarity, dim=-1)
        target_prob = ptu.eye(len(similarity))
        accuracy = (pred_prob * target_prob).sum(-1)
        diff = pred_prob-target_prob
    loss = (similarity*diff).sum(-1).mean()
    return loss, pred_prob, accuracy


class CPCModel(nn.Module, Processor):
    def __init__(
        self,
        env,
        processor,
        embedding_size=50,
        latent_size=128, 
        forward_layers=[256,256],
        reward_coef=1,
        alpha=0.1,
        cpc_coef=1,
        projector_layers=[],
        activation='relu',
    ):
        nn.Module.__init__(self)
        self.action_size = env.action_shape[0]
        self.embedding_size = embedding_size
        self.trunk = nn.Sequential(
            nn.Linear(processor.output_shape[0], self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.Tanh())
        self.projector_net = MLP(
            self.embedding_size,
            latent_size, 
            hidden_layers=projector_layers,
            activation=activation)
        self.predict_net = MLP(
            embedding_size+self.action_size,
            embedding_size+1,
            hidden_layers=forward_layers,
            activation=activation)
        self.forward_trunk = nn.Sequential(
            nn.LayerNorm(self.embedding_size),
            nn.Tanh())
        self.latent_size = latent_size 
        self.alpha = alpha
        self.cpc_coef = cpc_coef
        self.reward_coef = reward_coef

    def compute_auxiliary_loss(self, obs, a, r ,next_obs, 
            next_a, n_step=0, log=False, frame=None):
        e_size = self.embedding_size
        #reward 
        cat_obs = torch.cat([obs, next_obs],dim=0)
        cat_h = self.trunk(cat_obs)
        h, next_h = torch.chunk(cat_h, 2)
        pred_feature = torch.cat([h,a],dim=-1)
        pred_hr = self.predict_net(pred_feature)
        pred_next_h = self.forward_trunk(pred_hr[:,:e_size])
        pred_r = pred_hr[:,-1:]
        r_loss = F.mse_loss(pred_r, r.detach())
        #contrastive
        cat_h = torch.cat([next_h, pred_next_h], dim=0)
        cat_z = self.projector_net(cat_h)
        next_z, pred_next_z = torch.chunk(cat_z,2)
        next_z1, next_z2 = torch.chunk(next_z, 2)
        pred_next_z1, pred_next_z2 = torch.chunk(pred_next_z, 2)
        cl_loss1, pred_prob, accuracy = compute_cl_loss(next_z1, pred_next_z1, self.alpha)
        cl_loss2, _, _ = compute_cl_loss(next_z2, pred_next_z2, self.alpha)
        cl_loss = cl_loss1 + cl_loss2
        loss = self.reward_coef*r_loss + self.cpc_coef*cl_loss
        if log:
            logger.tb_add_histogram("aux/accuracy", accuracy, n_step)
            logger.tb_add_scalar("aux/cl_loss", cl_loss, n_step)
            logger.tb_add_scalar("aux/r_loss", r_loss, n_step)
            logger.tb_add_scalar("aux/mean_acc", accuracy.mean(), n_step)
            log_frame(frame, 0, 'origin', n_step, prefix="model/")
            pred_frame = torch.argmax(pred_prob[0])
            log_frame(frame, pred_frame, 'pred_frame', n_step, prefix="model/")
        return loss

    def process(self):
        raise NotImplementedError


class CURLModel(nn.Module, Processor):
    def __init__(
        self,
        env,
        processor,
        projector_layers=[256,256],
        embedding_size=50,
        latent_size=128, 
        alpha=0.1,
        activation='relu',
    ):
        nn.Module.__init__(self)
        self.action_size = env.action_shape[0]
        self.embedding_size = embedding_size
        self.trunk = nn.Sequential(
            nn.Linear(processor.output_shape[0], self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.Tanh())
        self.projector_net = MLP(
            self.embedding_size,
            latent_size, 
            hidden_layers=projector_layers,
            activation=activation)
        self.latent_size = latent_size 
        self.alpha = alpha

    def compute_auxiliary_loss(self, obs, a, r ,next_obs, 
            next_a, n_step=0, log=False, frame=None):
        #reward 
        h = self.trunk(obs)
        z = self.projector_net(h)
        z1, z2 = torch.chunk(z, 2)
        loss, pred_prob, accuracy = compute_cl_loss(z1, z2, self.alpha)
        if log:
            logger.tb_add_histogram("aux/accuracy", accuracy, n_step)
            logger.tb_add_scalar("aux/cl_loss", loss, n_step)
            logger.tb_add_scalar("aux/mean_acc", accuracy.mean(), n_step)
            log_frame(frame, 0, 'origin', n_step, prefix="model/")
            pred_frame = torch.argmax(pred_prob[0])
            log_frame(frame, pred_frame, 'pred_frame', n_step, prefix="model/")
        return loss

    def process(self):
        raise NotImplementedError