from collections import OrderedDict

import numpy as np
import torch
from torch.nn.modules import linear
from torch.nn.modules.sparse import Embedding
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from mirl.utils.process import Progress, Silent, format_for_process
import mirl.torch_modules.utils as ptu
from mirl.agents.sac_agent import SACAgent
from mirl.agents.drq.drq_agent import log_frame
from mirl.processors.utils import ImageAug, RandomShiftsAug
from mirl.utils.logger import logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pdb


import numpy as np
import copy
 
def traj_from_numpy(o):
    torch_o = {}
    for k in o:
        v = np.stack(o[k], axis=1)
        torch_o[k] = ptu.from_numpy(v)
    return torch_o

#TODO： check images的重构，现在cnn的输入没有减去0.5
class SLACAgent(SACAgent):
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        obs_processor,
        traj_processor,
        pretrain_steps=50000,
        pretrain_batch_size=64,
        kl_coef=1,
        reward_coef=1,
        multi_step_train=True,
        detach_posterior=False,
        conditioned_prior=False,
        constant_prior=False,
        train_latent_ratio=0.11111112, 
        independent_training=True,
        image_pad=4,
        image_aug=False,
        latent_lr=1e-4,
        silent=False,
        use_jit_script=True,
        analyze_embeddings=False,
        projection_lr=1e-4,
        **sac_agent_kwargs
    ):
        self.action_repeat = env.action_repeat
        super().__init__(
            env,
            policy,
            qf,
            qf_target,
            **sac_agent_kwargs
        )
        self.frame_stack = env.frame_stack
        self.latent_lr = latent_lr
        self.aug_trans = ImageAug(image_aug)
        self.aug_trans.add_trans(RandomShiftsAug(image_pad))
        self.latent_params = list(obs_processor.parameters()) + \
                list(traj_processor.parameters())
        self.latent_optimizer = self.optimizer_class(
            self.latent_params,
            lr=latent_lr,
            **self.optimizer_kwargs
        )
        if use_jit_script:
            traj_processor.bind_virtual()
            self.traj_processor = torch.jit.script(traj_processor) 
            self.obs_processor = torch.jit.script(obs_processor)
        else:
            self.traj_processor = traj_processor
            self.obs_processor = obs_processor
        self.use_jit_script = use_jit_script
        self.train_latent_ratio = train_latent_ratio
        # train latent coef
        self.kl_coef = kl_coef
        self.reward_coef = reward_coef
        self.multi_step_train = multi_step_train
        self.detach_posterior = detach_posterior
        self.conditioned_prior = conditioned_prior
        self.constant_prior = constant_prior
        # pretrain and train mode
        self.independent_training = independent_training
        self.pretrain_steps = pretrain_steps
        self.pretrain_batch_size = pretrain_batch_size
        self.progress_class = Silent if silent else Progress
        self.analyze_embeddings = analyze_embeddings
        if analyze_embeddings:
            assert env.return_state
            self._s_size = env.state_size
            # for latent input
            _e_size = traj_processor.output_shape[0]
            self.latent_linear_projection = nn.Linear(_e_size, self._s_size)
            self.latent_linear_projection.to(ptu.device)
            self.latent_projection_optimizer = self.optimizer_class(
                self.latent_linear_projection.parameters(),
                lr=projection_lr,
                **self.optimizer_kwargs
            )
            # for history input
            _e_size = policy.feature_size
            self.history_linear_projection = nn.Linear(_e_size, self._s_size)
            self.history_linear_projection.to(ptu.device)
            self.history_projection_optimizer = self.optimizer_class(
                self.history_linear_projection.parameters(),
                lr=projection_lr,
                **self.optimizer_kwargs
            )
            self._state_dim_name = []
            for k,v in env.state_dim_dict.items():
                for i in range(v):
                    dim_name = k + ("_%d"%i)
                    self._state_dim_name.append(dim_name)
            self._s_mean, self._s_std = None, None
            self._s_fac = 0.995
    
    def _get_step_feature(self, o):
        o = traj_from_numpy(o)
        batch_size = len(o['frames'])
        e_ = self.obs_processor(o['frames']).view(batch_size,-1)
        a_ = o['actions'].view(batch_size,-1)
        features = torch.cat( (e_, a_), dim=-1 )
        return features

    def step_init(self, o, **kwargs):
        n = o.shape[0] if hasattr(o, "shape") else 1
        shape = (n, *self.env.action_space.shape)
        action = np.random.randn(*shape) * 2
        action = np.tanh(action)
        return action, {}   

    def step_explore(self, o, **kwargs):
        o = self._get_step_feature(o)
        with self.policy.deterministic_(False):
            a, _ = self.policy.action(o, **kwargs)
        a = ptu.get_numpy(a)
        return a, {}

    def step_exploit(self, o, **kwargs):
        return self.step_explore(o, **kwargs)

    @property
    def num_init_steps(self):
        return self._num_init_steps * self.action_repeat

    @property
    def num_explore_steps(self):
        return self._num_explore_steps * self.action_repeat

    def pretrain(self, pool):
        progress = self.progress_class(self.pretrain_steps)
        for _ in range(self.pretrain_steps):
            progress.update()
            torch_batch = pool.random_batch_torch(self.pretrain_batch_size)
            params = self.train_from_torch_batch(torch_batch, True, 1)
            progress.set_description(format_for_process(params))

    def compute_q_target(self, actor_next_obs, next_obs, rewards, terminals, v_pi_kwargs={}):
        with torch.no_grad():
            alpha = self._get_alpha()
            next_action, next_policy_info = self.policy.action(
                actor_next_obs, **self.next_sample_kwargs
            )
            log_prob_next_action = next_policy_info['log_prob']
            target_q_next_action = self.qf_target.value(
                next_obs, 
                next_action, 
                **v_pi_kwargs
            )[0] - alpha * log_prob_next_action
            q_target = rewards + (1.-terminals)*self.discount*target_q_next_action
        return q_target

    def compute_latent_loss(
        self, 
        z1, z2,
        post_mean, post_std, 
        prior_mean, priot_std,
        frame, 
        action,
        reward,
        terminal,
        prefix='latent/'
    ):
        if self.detach_posterior:
            post_mean, post_std = post_mean.detach(), post_std.detach() 
        losses, frames = self.traj_processor.compute_latent_loss(
            z1, z2,
            post_mean, post_std, 
            prior_mean, priot_std,
            frame, 
            action,
            reward,
            0.0
        )
        loss_kld, loss_image, loss_reward = losses
        assert len(loss_kld.shape) == 2
        assert len(loss_image.shape) == 2
        assert len(loss_reward.shape) == 2
        loss_kld = loss_kld.sum(dim=1).mean()
        loss_image = loss_image.sum(dim=1).mean()
        loss_reward = loss_reward.sum(dim=1).mean()
        latent_loss = loss_kld*self.kl_coef + loss_image \
            +loss_reward*self.reward_coef
        latent_info = self.log_latent(losses, frames, loss_kld, 
            loss_image, loss_reward, latent_loss, prefix)
        return latent_loss, latent_info

    def log_latent(self, losses, frames, loss_kld, loss_image, 
        loss_reward, latent_loss, prefix):
        frame, pred_frame = frames
        loss_names = ["latent_loss_kld/", "latent_loss_image/", "latent_loss_reward/"]
        if self._log_tb_or_not():
            if logger.log_or_not(logger.WARNING):
                logger.tb_add_images('seq_vae/ground_truth', frame[0]+0.5, self.num_train_steps)
                logger.tb_add_images('seq_vae/posterior', pred_frame[0]+0.5, self.num_train_steps)
            if logger.log_or_not(logger.INFO):
                for loss, name in zip(losses, loss_names):
                    for i in range(loss.shape[1]):
                        _name = name+'step_%d'%i
                        logger.tb_add_scalar(_name+"/scale", loss[:,i].mean(), self.num_train_steps)
                        logger.tb_add_histogram(_name+"/dist", loss[:,i], self.num_train_steps)
            logger.tb_flush()
        if logger.log_or_not(logger.INFO):
            for loss, name in zip(losses, loss_names):
                self.debug_info[name] = loss
        # compute sttistics
        latent_info = OrderedDict()
        latent_info[prefix+'loss_kld'] = loss_kld.item()
        latent_info[prefix+'loss_image'] = loss_image.item()
        latent_info[prefix+'loss_reward'] = loss_reward.item()
        latent_info[prefix+'loss'] = latent_loss.item()
        return latent_info

    def log_latent_grad_norm(self):
        if self._log_tb_or_not():
            norm = 0
            for p in self.latent_params:
                if p.grad is not None:
                    norm = norm + p.grad.norm()
            logger.tb_add_scalar('norm/latent_grad', norm, self.num_train_steps)
        if self._log_tb_or_not() and logger.log_or_not(logger.INFO):
            self.debug_info['latent_grad_norm'] = norm.item()

    def log_frame_for_latent_training(self, frame):
        # plot frames for the max/min prediction bias
        if self._log_tb_or_not() and logger.log_or_not(logger.INFO):
            for k in self.debug_info:
                if k.startswith("latent_loss_"):
                    loss = self.debug_info[k].mean(-1)
                    max_loss, ind = torch.max(loss, dim=0)
                    logger.tb_add_scalar("max_"+k, max_loss, self.num_train_steps)
                    log_frame(frame, ind, "max_"+k, self.num_train_steps)
                    logger.tb_flush()

    def log_frame_for_critic_training(self, frame):
        # plot frames for the max/min prediction bias
        if self._log_tb_or_not() and logger.log_or_not(logger.INFO):
            max_diff, index = torch.max(self.debug_info["diff"], dim=1)
            for i, (md, ind) in enumerate(zip(max_diff, index)):
                logger.tb_add_scalar('diff/q%d_max'%i, md, self.num_train_steps)
                log_frame(frame, ind, 'diff/q%d_max'%i, self.num_train_steps)
            min_diff, index = torch.min(self.debug_info["diff"], dim=1)
            for i, (md, ind) in enumerate(zip(min_diff, index)):
                logger.tb_add_scalar('diff/q%d_min'%i, md, self.num_train_steps)
                log_frame(frame, ind, 'diff/q%d_min'%i, self.num_train_steps)
            logger.tb_flush()

    def train_from_torch_batch(self, batch, only_update_latent=False, train_latent_ratio=None):
        if train_latent_ratio is None:
            train_latent_ratio = self.train_latent_ratio
        assert batch['frames'].shape[1] == self.frame_stack+1
        for k in ['rewards', 'terminals', 'actions']:
            assert batch[k].shape[1] == self.frame_stack, k
        batch_size = len(batch['frames'])
        latent_n = int(train_latent_ratio*batch_size)
        f = batch['frames']
        a = batch['actions']
        r = batch['rewards']
        t = batch['terminals']
        self.log_batch(r, t)
        f_aug = self.aug_trans(f)
        #######################
        # update latent model #
        #######################
        f_1 = f[:latent_n]
        a_1 = a[:latent_n]
        r_1 = r[:latent_n]
        t_1 = t[:latent_n]
        e_1 = self.obs_processor(f_aug[:latent_n])
        if self.multi_step_train:
            z1, z2, post_mean, post_std = self.traj_processor.posterior_process(e_1, a_1)
            if self.conditioned_prior:
                prior_mean, priot_std = self.traj_processor.conditioned_prior_process(z1[:,0], a_1)
            elif self.constant_prior:
                prior_mean, priot_std = torch.zeros_like(post_mean), torch.ones_like(post_std)
            else:
                prior_mean, priot_std = self.traj_processor.prior_process(a_1)
        else:
            z1, z2, post_mean, post_std, prior_mean, priot_std = self.traj_processor.process(e_1, a_1)
        latent_loss, latent_info = self.compute_latent_loss(
            z1, z2, 
            post_mean, post_std, 
            prior_mean, priot_std,
            f_1, a_1, r_1, t_1
        )
        self.latent_optimizer.zero_grad()
        latent_loss.backward()
        self.log_latent_grad_norm()
        self.latent_optimizer.step()
        self.train_info.update(latent_info)
        self.log_frame_for_latent_training(f_1)
        if only_update_latent:
            self.log_train_info()
            self.num_train_steps += 1
            return self.train_info
        ###################################### for sac ######################################
        if self.independent_training:
            sac_n = batch_size - latent_n
        else:
            sac_n = batch_size
        # more efficient without computing gradient
        with torch.no_grad():
            f_2 = f[-sac_n:]
            a_2 = a[-sac_n:]
            r_2 = r[-sac_n:]
            t_2 = t[-sac_n:]
            e_2 = self.obs_processor(f_aug[-sac_n:])
            z1, z2, p_mean, p_std = self.traj_processor.posterior_process(e_2, a_2)
            z = torch.cat([z1, z2], -1)
            obs_actor = torch.cat( 
                [e_2[:,:-1].reshape(sac_n, -1), a_2[:,:-1].reshape(sac_n, -1)], 
                dim=-1)
            next_obs_actor = torch.cat(
                [e_2[:,1:].reshape(sac_n, -1), a_2[:,1:].reshape(sac_n, -1)], 
                dim=-1)
        ######################
        # analyze embeddings #
        ######################
        if self.analyze_embeddings:
            s_2 = batch['states'][-sac_n:]
            projection_loss = self._update_projection(z[:,-2,], z[:,-1], s_2[:,-2], s_2[:,-1], "latent")
            self.train_info['latent_to_state/loss'] = projection_loss.item()
            projection_loss = self._update_projection(obs_actor, next_obs_actor, s_2[:,-2], s_2[:,-1], "history")
            self.train_info['history_to_state/loss'] = projection_loss.item()

        #################
        # update critic #
        #################
        q_target = self.compute_q_target(
            next_obs_actor, 
            z[:,-1], 
            r_2[:,-1], 
            t_2[:,-1], 
            self.next_v_pi_kwargs
        )
        qf_loss, train_qf_info = self.compute_qf_loss(z[:,-2], a_2[:,-1], q_target)
        self.log_frame_for_critic_training(f_2)
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.log_critic_grad_norm()
        self.qf_optimizer.step()
        if self.num_train_steps % self.target_update_freq == 0:
            self._update_target(self.soft_target_tau)
        self.train_info.update(train_qf_info)
        ###########################
        # update actor and alpha #
        ###########################
        if self.num_train_steps % self.policy_update_freq == 0:
            new_action, action_info = self.policy.action(
                obs_actor, **self.current_sample_kwargs)
            # update alpha
            if self.use_automatic_entropy_tuning:
                alpha_loss, train_alpha_info = self.compute_alpha_loss(action_info)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.train_info.update(train_alpha_info)
            # update policy
            policy_loss, train_policy_info = self.compute_policy_loss(
                z[:,-2], 
                new_action, 
                action_info, 
                self.current_v_pi_kwargs
            )
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
        if logger.log_or_not(logger.DEBUG):
            self.anomaly_detection()
        batch_keys = list(batch.keys())
        for k in batch_keys:
            del batch[k]
        self.log_train_info()
        self.num_train_steps += 1
        return copy.deepcopy(self.train_info)

    def _update_projection(
        self, 
        cur_e,
        next_e, 
        cur_s,
        next_s,
        embedding_type="latent"
    ):
        batch_size = cur_e.shape[0]
        with torch.no_grad():
            e = torch.cat([cur_e, next_e], dim=0)
            s = torch.cat([cur_s, next_s], dim=0)
            #### normalize
            batch_mean = torch.mean(s, dim=0)
            batch_std = torch.std(s, dim=0) + 1e-6
            if self._s_mean is None:
                self._s_mean = batch_mean
                self._s_std = batch_std
            else:
                self._s_mean = self._s_fac*self._s_mean + (1-self._s_fac)*batch_mean
                self._s_std = self._s_fac*self._s_std + (1-self._s_fac)*batch_std
            normalized_s = (s-self._s_mean) / self._s_std
        # update
        if embedding_type == "latent":
            linear_projection = self.latent_linear_projection
            projection_optimizer = self.latent_projection_optimizer
        elif embedding_type == "history":
            linear_projection = self.history_linear_projection
            projection_optimizer = self.history_projection_optimizer
        normalized_pred_s = linear_projection(e.detach())
        projection_loss = F.mse_loss(normalized_pred_s, normalized_s.detach())
        projection_optimizer.zero_grad()
        projection_loss.backward()
        projection_optimizer.step()
        if self._log_tb_or_not() and logger.log_or_not(logger.INFO):
            pred_s = normalized_pred_s*self._s_std + self._s_mean
            pred_s = pred_s.view(2, batch_size, self._s_size)
            pred_s = pred_s.detach().cpu().numpy()
            s = s.view(2, batch_size, self._s_size)
            s = s.detach().cpu().numpy()
            abs_diff = np.abs(pred_s - s)
            prefix = embedding_type + "_to_state"
            for i, name in enumerate(self._state_dim_name):
                _prefix = prefix + '-' + name
                _s = s[:,:,i]
                _pred_s = pred_s[:,:,i]
                _abs_diff = abs_diff[:,:,i]
                min_x, max_x = _s.min(), _s.max()
                fig, ax = plt.subplots()
                ax.grid(ls='--')
                ax.plot([min_x,max_x],[min_x,max_x], color='gray', ls='--', alpha=0.5)
                for t in range(2):
                    label = "cur_state" if t == 0 else "next_state"
                    ax.scatter(_s[t,:], _pred_s[t,:], alpha=0.2, label=label)
                    _fig, _ax = plt.subplots()
                    _ax.grid(ls='--')
                    _ax.plot([min_x,max_x],[min_x,max_x], color='gray', ls='--', alpha=0.5)
                    _ax.scatter(_s[t,:], _pred_s[t,:], alpha=0.2)
                    logger.tb_add_figure("%s/%s"%(_prefix, label), _fig, self.num_train_steps)
                ax.legend()
                logger.tb_add_figure("%s/all"%_prefix, fig, self.num_train_steps)
                _prefix = prefix + '/' + name
                logger.tb_add_scalar("%s/abs_diff_scalar"%_prefix, _abs_diff[:,:,].mean(), self.num_train_steps)
                logger.tb_add_histogram("%s/abs_diff_histogram"%_prefix, _abs_diff, self.num_train_steps)
                logger.tb_flush()
        return projection_loss

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.qf_target,
            self.obs_processor,
            self.traj_processor
        ]

    def get_snapshot(self):
        if self.use_jit_script: # torch.save can not save jit.script_module
            return dict(
                policy=self.policy,
                qf=self.qf,
                qf_target=self.qf_target
            )
        else:
            return dict(
                policy=self.policy,
                qf=self.qf,
                qf_target=self.qf_target,
                obs_processor=self.obs_processor,
                traj_processor=self.traj_processor
            )

    def anomaly_detection(self):
        if self.debug_info['qf_loss'].item() > self.debug_loss_threshold:
            self.plot_value_scatter(prefix='anomaly/')
            self.anomaly_count += 1
            if self.anomaly_count % 10000 == 0 :
                pdb.set_trace()
            anomaly = True
        else:
            anomaly = False
        info_keys = list(self.debug_info.keys())
        for k in info_keys:
            del self.debug_info[k]
        return anomaly
        
