from mirl.agents.td3_agent import TD3Agent
import mirl.torch_modules.utils as ptu
import torch.nn.functional as F
import copy
from mirl.utils.logger import logger


class TD3BCAgent(TD3Agent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        pool=None,
        normalize_obs=True,
        bc_alpha=2.5,
        **td3_kwargs
    ):
        super().__init__(
            env,
            policy,
            qf,
            qf_target,
            **td3_kwargs
        )
        self.bc_alpha = bc_alpha
        self.normalize_obs = normalize_obs
        if normalize_obs:
            assert pool is not None
            assert pool.compute_mean_std
            self.obs_mean, self.obs_std = pool.get_mean_std()['observations']
            self.obs_mean = ptu.from_numpy(self.obs_mean)
            self.obs_std = ptu.from_numpy(self.obs_std)

    def process_obs(self, obs):
        if self.normalize_obs:
            obs = obs = (obs-self.obs_mean)/(self.obs_std+1e-6)
        return obs
    
    def step_explore(self, o, **kwargs):
        raise NotImplementedError

    def step_exploit(self, o, **kwargs):
        with self.policy.deterministic_(True):
            o = ptu.from_numpy(o)
            o = self.process_obs(o)
            a,_ = self.policy.action(o, **kwargs)
            a = ptu.get_numpy(a)
        return a, {}

    def compute_policy_loss(
        self, obs, new_action, 
        origin_action, v_pi_kwargs={}
    ):
        q_new_action, _ = self.qf.value(
            obs, 
            new_action, 
            return_ensemble=False, 
            **v_pi_kwargs
        )
        q_pi_mean = q_new_action.mean()
        policy_loss = -q_pi_mean
        policy_info = self._log_policy_info(
            new_action, policy_loss, q_pi_mean)
        ### for td3bc ###
        bc_lambda = self.bc_alpha / q_pi_mean.detach().abs().mean()
        bc_loss = F.mse_loss(new_action, origin_action)
        policy_loss = policy_loss*bc_lambda + bc_loss
        if self._log_tb_or_not():
            logger.tb_add_scalar("policy/bc_loss", bc_loss, self.num_train_steps)
        ###### end ######
        return policy_loss, policy_info

    def train_from_torch_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        obs = self.process_obs(obs)
        next_obs = self.process_obs(next_obs)
        self.log_batch(rewards, terminals)
        ################
        # update critic #
        ################
        q_target = self.compute_q_target(next_obs, rewards, terminals, self.next_v_pi_kwargs)
        qf_loss, train_qf_info = self.compute_qf_loss(obs, actions, q_target)
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.log_critic_grad_norm()
        self.qf_optimizer.step()
        if self.num_train_steps % self.target_update_freq == 0:
            self._update_target(self.soft_target_tau)
        self.train_info.update(train_qf_info)
        ################
        # update actor #
        ################
        if self.num_train_steps % self.policy_update_freq == 0:
            new_action, action_info = self.policy.action(obs, **self.current_sample_kwargs)
            policy_loss, train_policy_info = self.compute_policy_loss(
                obs, 
                new_action, 
                actions,  #NOTE: origin action for bc
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
        self.log_train_info()
        self.num_train_steps += 1
        return copy.deepcopy(self.train_info)