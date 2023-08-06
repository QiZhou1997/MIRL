import numpy as np
import warnings
from collections import OrderedDict

from numpy.core.shape_base import block

from mirl.pools.utils import get_batch, _random_batch_independently, _shuffer_and_random_batch
from mirl.pools.trajectory_pool import TrajectoryPool
import mirl.torch_modules.utils as ptu
from mirl.utils.logger import logger
import random
import warnings
from termcolor import colored
import copy
import numpy as np
import warnings
from collections import OrderedDict

from mirl.pools.utils import get_batch, _random_batch_independently, _shuffer_and_random_batch
from mirl.pools.trajectory_pool import TrajectoryPool
from mirl.pools.extended_trajectory_pool import ExtendedTrajectoryPool
import mirl.torch_modules.utils as ptu
from mirl.utils.logger import logger
import copy


# TODO:check bug
class DomainPool(TrajectoryPool):
    def __init__(
        self, 
        env, 
        max_size=1e6, 
        traj_len=None,  
        return_traj=None,
        compute_mean_std=False,
        ndomain_per_batch=0,
    ):
        super().__init__(env, max_size, traj_len, return_traj, compute_mean_std)
        self.ndomain_per_batch = ndomain_per_batch
        self._domain_index = {}
        self._domain_index_start = {}
        self._domain_index_stop = {}
        self._domain_size = {}
    
    def update_index(self):
        self._path_len += 1
        # valid traj if its length is larger than _traj_len
        if self._path_len >= self.traj_len:
            self._index[self._index_stop] = self._stop 
            self._index_stop = (self._index_stop + 1) % self.max_size
            self._size += 1
            domain_index_stop = self._domain_index_stop[self._cur_domain]
            domain_index = self._domain_index[self._cur_domain]
            domain_index[domain_index_stop] = self._stop
            self._domain_index_stop[self._cur_domain] = (domain_index_stop+1)%self.max_size
            self._domain_size[self._cur_domain] += 1
            
        # drop invalid index
        # for example: _stop=0, traj_len=3, drop 2=0+3-1 
        first_index = self._index[self._index_start]
        if self._size>0 and first_index==(self._stop+self.traj_len-1) % self.max_size:
            self._index_start = (self._index_start + 1) % self.max_size
            self._size -= 1
            for domain, domain_index_start in self._domain_index_start.items():
                domain_first_index = self._domain_index[domain][domain_index_start]
                domain_size = self._domain_size[domain]
                if domain_size > 0 and first_index == domain_first_index:
                    self._domain_index_start[domain] = (domain_index_start+1)%self.max_size
                    self._domain_size[domain] -= 1

        # update _stop
        self._stop = (self._stop + 1) % self.max_size

    def set_domain(self, domain):
        self._cur_domain = domain
        if domain not in self._domain_index:
            self._domain_index[domain] = np.zeros((int(self.max_size),), dtype=int)
            self._domain_index_start[domain] = 0
            self._domain_index_stop[domain] = 0
            self._domain_size[domain] = 0

    def add_samples(self, samples):
        self.set_domain(samples['env_infos']['domain'])
        super().add_samples(samples)
        
    def get_batch_random_index(self, batch_size):
        if self.ndomain_per_batch < 1:
            return TrajectoryPool.get_batch_random_index(self, batch_size)
        assert batch_size%self.ndomain_per_batch == 0
        block_size = batch_size / self.ndomain_per_batch
        domains = []
        n = 0
        for k, s in self._domain_size.items():
            if s  >= block_size:
                n += 1
                domains.append(k)
        if n < self.ndomain_per_batch:
            return TrajectoryPool.get_batch_random_index(self, batch_size)
        domains = random.sample(domains, n)
        ind_list = []
        for domain in domains:
            ind_ind = np.random.randint(0, self._domain_size[domain], block_size)
            ind_ind = (ind_ind + self._domain_index_start[domain]) % self.max_size
            ind_list.append(self._domain_index[domain][ind_ind])
        ind = np.concatenate(ind_list)
        return ind


# TODO:check bug
class HardNegPool(TrajectoryPool):
    def __init__(
        self, 
        env, 
        max_size=1e6, 
        traj_len=None,  
        return_traj=None,
        compute_mean_std=False,
        k_seed=8,
        range=32
    ):
        super().__init__(env, max_size, traj_len, return_traj, compute_mean_std)
        self.k_seed = k_seed
        self.range = range

    def get_batch_random_index(self, batch_size):
        seeds = np.random.randint(0, self._size, self.k_seed)
        ind_ind = []
        each_n = batch_size // self.k_seed
        for i in range(self.k_seed):
            sub = np.random.permutation(self.range)[:each_n]
            sub = seeds[i] + sub
            ind_ind.append(sub)
        ind_ind = np.concatenate(ind_ind)
        ind_ind = ind_ind % self._size
        ind_ind = (ind_ind + self._index_start) % self.max_size
        ind = self._index[ind_ind]
        # ind_ind = np.random.randint(0, self._size, batch_size)
        # ind_ind = (ind_ind + self._index_start) % self.max_size
        # ind = self._index[ind_ind]
        # return ind
        return ind


class ExtendedDomainPool(ExtendedTrajectoryPool, DomainPool):
    def __init__(
        self, 
        env, 
        max_size=1e6, 
        traj_len=None,  
        return_traj=None,
        compute_mean_std=False,
        ndomain_per_batch=0,
    ):
        ExtendedTrajectoryPool.__init__(
            self, env, max_size, 
            traj_len=traj_len, 
            return_traj=return_traj, 
            compute_mean_std=compute_mean_std
        )
        self.ndomain_per_batch = ndomain_per_batch
        # for domain sample
        self._domain_index = {}
        self._domain_index_start = {}
        self._domain_index_stop = {}
        self._domain_size = {}
        # for noise label
        self._num_domain = 0
        self._domain_label_dict = {}
        self._total_noise_size = 0
        self._noise_label_start = {}

    def _get_default_fields(self):
        fields = super()._get_default_fields()
        fields['states'] = {'shape': self._state_shape, 'type': np.float32}
        fields['domain_labels'] = {'shape': (1,), 'type': int}
        fields['noise_labels'] = {'shape': (1,), 'type': int}
        return fields

    def set_domain(self, domain, domain_size):
        self._cur_domain = domain
        if domain not in self._domain_index:
            # for domain sample
            self._domain_index[domain] = np.zeros((int(self.max_size),), dtype=int)
            self._domain_index_start[domain] = 0
            self._domain_index_stop[domain] = 0
            self._domain_size[domain] = 0
            # for noise label
            self._domain_label_dict[domain] = self._num_domain
            self._num_domain += 1
            self._noise_label_start[domain] = self._total_noise_size
            self._total_noise_size += domain_size
        domain_label = self._domain_label_dict[domain]
        self._cur_domain_label = np.array([[domain_label]], dtype=int)

    def _get_noise_label(self, info):
        domain = info["domain"]
        noise_label = info["noise_label"]
        noise_label += self._noise_label_start[domain]
        next_noise_label = info["next_noise_label"]
        next_noise_label += self._noise_label_start[domain]
        self.next_noise_label = next_noise_label
        return np.array([[noise_label]], dtype=int)

    # only support add_single_sample
    def add_samples(self, samples):
        info = samples['env_infos']
        self.set_domain(info['domain'], info['domain_size'])
        one_step_sample = {}
        for k in self.fields:
            if k in samples:
                one_step_sample[k] = samples[k]
        o = samples["observations"]
        one_step_sample["frames"] = self.get_last_frame(o) 
        one_step_sample["states"] = info["state"]
        self._next_s = info["next_state"]
        one_step_sample["domain_labels"] = self._cur_domain_label
        one_step_sample["noise_labels"] = self._get_noise_label(info)
        self._add_one_step_sample(one_step_sample)
        return 1

    def end_a_path(self, next_o):
        one_step_sample = {}
        one_step_sample['frames'] = self.get_last_frame(next_o) 
        one_step_sample['states'] = self._next_s
        one_step_sample['noise_label'] = self.next_noise_label
        one_step_sample['domain_labels'] = self._cur_domain_label
        self._add_one_step_sample(one_step_sample)

    def random_batch_torch(self, batch_size, keys=None, without_keys=[]):
        index = self.get_batch_random_index(batch_size)
        traj_index = self.get_traj_index(index)
        if self.return_traj:
            torch_batch = {}
            for k in self.fields:
                data = self.get_traj_by_index(traj_index, k)
                if k in ['domain_labels', 'noise_labels']:
                    data = ptu.LongTensor(data)
                else:
                    data = ptu.from_numpy(data)
                if k in ['frames', 'states', 'domain_labels', 'noise_labels']: #traj_len
                    torch_batch[k] = data  
                else: #traj_len -1
                    torch_batch[k] = data[:,:-1] # Note!
            return torch_batch
        else:
            torch_batch = {} 
            frame = self.get_traj_by_index(traj_index, 'frames')
            frame = ptu.from_numpy(frame)  # (B, traj_len, channel, w, h)
            frame = frame.view(-1, self.traj_len*self._channel, *self._frame_shape[1:])
            channel = self.frame_stack*self._channel
            torch_batch['observations'] = frame[:,:channel]
            torch_batch['next_observations'] = frame[:,-channel:]
            torch_batch["next_domain_labels"] = ptu.LongTensor(self.dataset['domain_labels'][index])
            torch_batch["next_noise_labels"] = ptu.LongTensor(self.dataset['noise_labels'][index])
            torch_batch['next_states'] = ptu.from_numpy(self.dataset['states'][index])
            index = (index - 1) % self.max_size
            for k in ['actions', 'rewards', 'terminals', 'states']:
                torch_batch[k] = ptu.from_numpy(self.dataset[k][index])
            for k in ['domain_labels', 'noise_labels']:
                torch_batch[k] = ptu.LongTensor(self.dataset[k][index])
            return torch_batch


if __name__ == "__main__":
    from mirl.environments.dmc_env import DMControlEnv
    from mirl.environments.continuous_vector_env import ContinuousVectorEnv
    from mirl.agents.base_agent import Agent
    from mirl.collectors.step_collector import SimpleCollector
    import pdb
    # env = ContinuousVectorEnv("cheetah",4)
    env_kwargs = dict(
        domain="cartpole", 
        task="swingup", 
        distracting="background",
        dynamic_distracting=True,
        background_kwargs= {"num_videos": 2},
        return_state=True,
        return_traj=False,#False
        action_repeat=8,
        obs_before_reset="repeat", 
        record_video=False,
        env_render_size=480,
        render_via_env=True,
        video_fps=10,
        video_dir='/home/qizhou',
        video_prefix='background_video_',
    )
    env = DMControlEnv(**env_kwargs)
    pi = Agent(env)
    pool = ExtendedDomainPool(env, 10000, ndomain_per_batch=-1)
    cl = SimpleCollector(env, pi, pool)
    for i in range(2):
        for i in range(125):
            cl.collect_new_steps(3,125,step_mode='init')
        print(
        pool._domain_index,
        pool._domain_index_start,
        pool._domain_index_stop,
        pool._domain_size,
        pool._noise_label_start)
    print(pool.random_batch_torch(8))
    # # print(pool.__dict__)
    # np.save(open('/home/qiz/test_pool.npy','wb'), pool, allow_pickle=True)
    # newpool = np.load(open('/home/qiz/test_pool.npy','rb'), allow_pickle=True).item()
    # # print(newpool.__dict__)
    # print(type(newpool))
    # print([k for k in pool.__dict__ if not callable(pool.__dict__[k])])
    # pickle.dump(pool, open('/home/qiz/test_pool.pkl','wb'))
    # newpool = pickle.load(open('/home/qiz/test_pool.npy','rb')).item()

    import pdb
    pdb.set_trace()
    
