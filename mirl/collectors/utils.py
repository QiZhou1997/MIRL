import numpy as np
import torch
from mirl.utils.misc_untils import combine_item
import mirl.torch_modules.utils as ptu
from collections import OrderedDict
from tqdm import tqdm

def rollout(
        env,
        agent,
        max_time_steps=1000,
        use_tqdm=False,
    ):
    
    pass

# Note the difference between agent_infos/env_infos
# TODO: check 
class Path():
    def __init__(self, data_type="numpy"):
        self.actions = [] # len
        self.agent_infos = [] # len; [{[],[]}, {[],[]}]
        self.observations = [] # len + 1
        self.rewards = [] # len
        self.terminals = [] # len + 1
        self.env_infos = [] # len; [[{}, {}], [{}, {}]]
        self.extra_info = []

        self.path = None
        self.batch = None
        self.step_count = 0

        self.data_type = data_type

    def start_new_paths(self, o , t=None):
        self.observations.append(o)
        self.o = o
        self.n_env = len(o)
        if self.data_type == "numpy":
            self.t = np.ones((self.n_env,1))
        elif self.data_type == "torch":
            self.t = ptu.ones
        self.terminals.append(self.t)
        
        #padding
        if len(self.actions) > 0:
            self.actions.append(self.actions[-1])
            self.agent_infos.append(self.agent_infos[-1])
            self.rewards.append(self.rewards[-1])
            self.env_infos.append(self.env_infos[-1])
        

    def update(self, a, agent_info, next_o, r, d, env_info):
        self.actions.append(a)
        self.agent_infos.append(agent_info)
        
        self.o = next_o
        if self.data_type == "numpy":
            self.step_count = self.step_count + (1-self.t.astype(int))
            self.t = np.logical_or(self.t, d)
        elif self.data_type == "torch":
            self.step_count = self.step_count + (1-self.t.int())
            self.t = torch.logical_or(self.t, d)

        self.observations.append(next_o)
        self.rewards.append(r)
        self.terminals.append(self.t)
        self.env_infos.append(env_info)
        return self.t
    
    def wipe_memory(self):
        self.__init__(self.data_type)
    
    def get_terminal(self):
        return self.t
    
    def get_total_steps(self):
        if self.data_type == "numpy":
            return np.sum(self.step_count)
        elif self.data_type == "torch":
            return torch.sum(self.step_count)

    # TODO：将那个函数移动到这
    def get_statistic(
        self,
        discard_uncomplete_paths=True,
    ):
        pass
    
    def get_useful_infos(
        self,
        info=[],
        useful_keys=None,
        return_type="path"
    ):
        raise NotImplementedError

    def get_batch(
        self,
        useful_env_info=None,
        useful_agent_info=None,
        useful_extra_info=None,
        qlearning_form=True,
    ):
        raise NotImplementedError

    def get_path(
        self,
        useful_env_info=None,
        useful_agent_info=None,
        useful_extra_info=None,
        cut_path=False
    ):
        raise NotImplementedError

def path_to_samples(paths):
    path_number = len(paths)
    data = paths[0]
    for i in range(1,path_number):
        data = combine_item(data, paths[i])
    return data


# def get_single_path_info(info, index):
#     single_path_info = {}
#     for k in info:
#         single_path_info[k] = info[k][:,index,:]
#     return single_path_info

# def split_paths(paths):
#     new_paths = []
#     for i in range(len(paths['actions'][0])):
#         path = dict(
#             observations=paths['observations'][:,i,:],
#             actions=paths['actions'][:,i,:],
#             rewards=paths['rewards'][:,i,:],
#             next_observations=paths['next_observations'][:,i,:],
#             terminals=paths['terminals'][:,i,:],
#             agent_infos=get_single_path_info(paths['agent_infos'],i),
#             env_infos=get_single_path_info(paths['env_infos'],i),
#         )
#         new_paths.append(path)
#     return new_paths

# def cut_path(path, target_length):
#     new_path = {}
#     for key, value in path.items():
#         if type(value) in [dict, OrderedDict]:
#             new_path[key] = cut_path(value, target_length)
#         else:
#             new_path[key] = value[:target_length]
#     return new_path



if __name__ == "__main__":
    # 分别测试torch，numpy。
    # 一个完整的path，不完整的，一个半，两个半
    # 测试不同的GAE
    # 测试statistic    
    # 存在或不存在terminal的环境 
    # for fake_env
    pass


# #TODO: check
# def compute_target(
#     self, 
#     value_function=None,
#     policy=None,
#     gamma=0.99,
#     lam=0.995,
#     mode="vtrace",
#     on_policy=False,
# ):
#     # mode=retrace or vtrace
#     # extra info
#     if mode == "vtrace":
#         v = []
#         for o in self.observations:
#             if self.data_type == "numpy":
#                 v.append(
#                     value_function.value_np(o)[0]
#                 )
#             elif self.data_type == "torch":
#                 v.append(
#                     value_function.value(o)[0]
#                 )
#         path_len = len(self.actions)
#         adv = []
#         first_adv = 0
#         for i in range(path_len,0,-1):
#             live = 1 - self.terminals[i]
#             if on_policy:
#                 phi = 1
#             else:
#                 raise NotImplementedError
#             coef = live*gamma
#             delta = self.rewards[i-1] + phi*coef*v[i] - v[i-1]
#             first_adv = delta+lam*coef*first_adv
#             adv.insert(0, first_adv)
#         self.extra_info["adv"] = adv
#     else:
#         raise NotImplementedError