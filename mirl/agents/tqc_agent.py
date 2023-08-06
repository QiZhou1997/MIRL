import torch
import mirl.torch_modules.utils as ptu
from mirl.agents.sac_agent import SACAgent

def quantile_huber_loss_f(samples, ensemble_atoms):
    pairwise_delta = samples[None, :, :, None] - ensemble_atoms[..., None, :]  #  batch x nets x samples x quantiles 
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)
    n_quantiles = ensemble_atoms.shape[-1]
    tau = ptu.arange(n_quantiles).float() / n_quantiles + 0.5 / n_quantiles
    loss = (torch.abs( tau - (pairwise_delta<0).float() ) * huber_loss).mean()
    return loss

class TQCAgent(SACAgent): 
    def __init__(
        self,
        env,
        policy,
        qf,
        qf_target,
        next_v_pi_kwargs={
            "percentage_drop":8,
        }, 
        current_v_pi_kwargs={
            "number_drop":0,
        },
        **sac_kwargs
    ):
        super().__init__(
            env, 
            policy, 
            qf, 
            qf_target, 
            next_v_pi_kwargs=next_v_pi_kwargs,
            current_v_pi_kwargs=current_v_pi_kwargs,
            **sac_kwargs
        )
    
    def compute_q_target(self, next_obs, rewards, terminals, v_pi_kwargs={}):
        with torch.no_grad():
            alpha = self._get_alpha()
            next_action, next_policy_info = self.policy.action(
                next_obs, **self.next_sample_kwargs
            )
            log_prob_next_action = next_policy_info['log_prob']
            _, info = self.qf_target.value(
                next_obs, 
                next_action, 
                return_atoms=True,
                **v_pi_kwargs
            )
            value_atoms = info['value_atoms']
            target_atoms_next_action = value_atoms - alpha * log_prob_next_action
            atoms_target = rewards + (1.-terminals)*self.discount*target_atoms_next_action
        return atoms_target

    def compute_qf_loss(self, obs, actions, atoms_target):
        _, value_info = self.qf.value(obs, actions, only_return_ensemble=True)
        q_atoms_ensemble = value_info['ensemble_atoms']
        qf_loss = quantile_huber_loss_f(atoms_target.detach(), q_atoms_ensemble)
        q_target_mean = torch.mean(atoms_target, dim=-1, keepdim=True)
        ensemble_mean = torch.mean(q_atoms_ensemble, dim=-1, keepdim=True)
        qf_info = self._log_q_info(q_target_mean, ensemble_mean, qf_loss)
        return qf_loss, qf_info    