from mirl.analyzers.base_analyzer import Analyzer
from mirl.environments.continuous_vector_env import ContinuousVectorEnv
from mirl.utils.eval_util import create_stats_ordered_dict
import numpy as np
import random
from tqdm import tqdm


def eval_state(env, 
               agent, 
               sim_data, 
               action, 
               episode_length, 
               discount, 
               num_paths,
               alpha=0,
               batch_size=256):
               
    print("evaluate state.....")

    env_name = env.env_name
    sim_data = sim_data * num_paths
    if action is not None:
        action = np.tile(action, (num_paths,1))
    total_number = len(sim_data)
    
    coe = np.ones((1,episode_length))
    for i in range(1,episode_length):
        coe[0,i] = coe[0,i-1] * discount
    all_ret = []

    for i in range(0,total_number,batch_size):
        cur_size = min(batch_size, total_number-i)
        print("total: %d\t current: %d"%(total_number, i))
        env = ContinuousVectorEnv(env_name, cur_size, asynchronous=False)
        env.set_state(sim_data[i:i+cur_size])

        rews = np.zeros((cur_size, episode_length))
        o = env.state_vector()
        live = np.ones((cur_size,1))
        for i in tqdm(range(episode_length)):
            if i == 0 and action is not None:
                a = action[i:i+cur_size]
                extra_r = 0
            else:
                a,info = agent.step(o, return_log_prob=True)
                log_prob = info["log_prob"]
                extra_r = - alpha * log_prob

            o,r,d,_ = env.step(a)
            r = r + extra_r
            live = live * (1-d)
            rews[:,i:i+1] = live*r
            if live.sum() < 1e-6: 
                break

        ret = np.sum(rews*coe, axis=-1)
        all_ret.append(ret)

    ret = np.concatenate(all_ret).reshape(num_paths,-1)
    env.close()
    return ret.mean(axis=0), ret.std(axis=0)

def get_sim_data(env, agent, sample_number, episode_length=1000, min_total_number=4000):
    print("select evaluation state...")
    n_env = int(min_total_number / episode_length)
    env = ContinuousVectorEnv(env.env_name, n_env, asynchronous=False)
    
    data = []
    total_number = 0
    live = np.ones((n_env,1))
    o = env.reset()
    while total_number < min_total_number:
        a,_ = agent.action_np(o)
        for i,l in enumerate(live):
            temp_sim_data = env.get_state()
            if l>0:
                data.append((temp_sim_data[i],o[i],a[i]))

        o,r,d,_ = env.step(a)
        live = live * (1-d)
        total_number += live.sum()
        if live.sum() < 1e-6: 
            break
    data = random.sample(data, sample_number)
    sim_data, obs, action = list(zip(*data))
    obs, action = np.array(obs), np.array(action)
    return sim_data, obs, action
    


class BiasAnalyzer(Analyzer):
    def __init__(self, 
                 env, 
                 policy, 
                 q_value, 
                 agent,
                 gamma=0.99,
                 eval_number=32,
                 episode_length=1000,
                 min_rollout=4000,
                 eval_paths=16
                 ):
        # TODO online and offline evaluator
        self.env = env 
        self.policy = policy
        self.q_value = q_value
        self.agent = agent
        self.gamma = gamma
        self.eval_number = eval_number
        self.episode_length = episode_length
        self.min_rollout = min_rollout
        self.eval_paths = eval_paths
        self.statis = {}
    
    def analyze(self,):
        alpha = self.agent.log_alpha.exp().item()

        sim_data, obs, action = get_sim_data(self.env, 
                                             self.policy, 
                                             self.eval_number, 
                                             self.episode_length, 
                                             self.min_rollout) 

        real_ret_mean, real_ret_std = eval_state(self.env, 
                                                self.policy, 
                                                sim_data, 
                                                action, 
                                                self.episode_length, 
                                                self.gamma, 
                                                self.eval_paths,
                                                alpha)

        results = {}
        results['Real Q'] = np.array(real_ret_mean)
        results["Pred Q"], _ = self.q_value.value_np(obs, action)
        assert len(results['Real Q']) == len(results["Pred Q"])
        results["Q Bias"] = results['Pred Q'] - results['Real Q'] 
        results["Bias Ratio"] = results["Q Bias"]  / (results['Real Q'].std()+1e-6)
        results["Bias Abs"] = np.abs(results["Q Bias"])
        statis = {}
        for k,v in results.items():
            self.statis.update(create_stats_ordered_dict(k, v))
        return self.statis

    def get_diagnostics(self):
        return self.statis

