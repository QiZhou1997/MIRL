import numpy as np
import warnings
from collections import OrderedDict

from mirl.pools.utils import get_batch, _random_batch_independently, _shuffer_and_random_batch
from mirl.pools.trajectory_pool import TrajectoryPool
import mirl.torch_modules.utils as ptu
from mirl.utils.logger import logger
import copy

# DrQ and SLAC
# o: len + 1; a: len; r: len; d: len

#frame: (pad, pad, pad, o_0, o_1, ..., o_N, o_(N+1), pad, pad, pad, o_0)
#action: (pad, pad, pad, a_0, a_1, ..., a_N, pad, pad, pad, pad, a_0)
class ExtendedTrajectoryPool(TrajectoryPool):
    def __init__(
        self, 
        env, 
        max_size=1e6, 
        **pool_kwargs
    ):
        self._state_shape = env.state_shape
        super().__init__(env, max_size=max_size, **pool_kwargs)


    def _get_default_fields(self):
        fields = super()._get_default_fields()
        fields['states'] = {'shape': self._state_shape, 'type': np.float32}
        return fields
    
    # only support add_single_sample
    def add_samples(self, samples):
        one_step_sample = {}
        for k in self.fields:
            if k in samples:
                one_step_sample[k] = samples[k]
        o = samples["observations"]
        one_step_sample["frames"] = self.get_last_frame(o) 
        one_step_sample["states"] = samples["env_infos"]["state"]
        self._next_s = samples["env_infos"]["next_state"]
        self._add_one_step_sample(one_step_sample)
        return 1

    def end_a_path(self, next_o):
        one_step_sample = {}
        one_step_sample['frames'] = self.get_last_frame(next_o) 
        one_step_sample['states'] = self._next_s
        self._add_one_step_sample(one_step_sample)

    def random_batch_torch(self, batch_size, keys=None, without_keys=[]):
        index = self.get_batch_random_index(batch_size)
        traj_index = self.get_traj_index(index)
        if self.return_traj:
            torch_batch = {}
            for k in self.fields:
                data = self.get_traj_by_index(traj_index, k)
                batch = ptu.from_numpy(data)
                if k == "frames" or k == "states": #traj_len
                    torch_batch[k] = batch  
                else: #traj_len -1
                    torch_batch[k] = batch[:,:-1] # Note!
            return torch_batch
        else:
            torch_batch= {} 
            frame = self.get_traj_by_index(traj_index, 'frames')
            frame = ptu.from_numpy(frame)  # (B, traj_len, channel, w, h)
            frame = frame.view(-1, self.traj_len*self._channel, *self._frame_shape[1:])
            channel = self.frame_stack*self._channel
            torch_batch['observations'] = frame[:,:channel]
            torch_batch['next_observations'] = frame[:,-channel:]
            torch_batch['next_states'] = ptu.from_numpy(self.dataset['states'][index])
            index = (index - 1) % self.max_size
            for k in ['actions', 'rewards', 'terminals', 'states']:
                torch_batch[k] = ptu.from_numpy(self.dataset[k][index])
            return torch_batch


if __name__ == "__main__":
    from mirl.environments.dmc_env import DMControlEnv
    from mirl.environments.continuous_vector_env import ContinuousVectorEnv
    from mirl.agents.base_agent import Agent
    from mirl.collectors.step_collector import SimpleCollector
    import pdb
    # env = ContinuousVectorEnv("cheetah",4)
    test = 2
    if test == 1:
        env = DMControlEnv("cheetah","run",8,frame_stack=10,return_traj=True,return_state=True, obs_before_reset="zero")
        pi = Agent(env)
        pool = ExtendedTrajectoryPool(env, 200)
        cl = SimpleCollector(env, pi, pool)
        for i in range(2):
            for i in range(25):
                cl.collect_new_steps(5,125,step_mode='init')
        # # print(pool.__dict__)
        # np.save(open('/home/qiz/test_pool.npy','wb'), pool, allow_pickle=True)
        # newpool = np.load(open('/home/qiz/test_pool.npy','rb'), allow_pickle=True).item()
        # # print(newpool.__dict__)
        # print(type(newpool))
        # print([k for k in pool.__dict__ if not callable(pool.__dict__[k])])
        # pickle.dump(pool, open('/home/qiz/test_pool.pkl','wb'))
        # newpool = pickle.load(open('/home/qiz/test_pool.npy','rb')).item()
            print('test')
            print(pool.dataset['frames'][:16,0,31,30:40])
            print(pool.fields)
            print(pool.dataset['states'][:16])
            print(pool._index)
            print(pool.get_batch_random_index(125))
            batch = pool.random_batch_torch(20)
            print(batch['frames'][:,:,0,31,30:40])
            print(batch['rewards'])
            print(batch['actions'])
            print(batch['terminals'])
            print(batch['states'])
    if test == 2:
        env = DMControlEnv("cheetah","run",8,frame_stack=10,return_traj=False,return_state=True, obs_before_reset="zero")
        pi = Agent(env)
        pool = ExtendedTrajectoryPool(env, 200)
        cl = SimpleCollector(env, pi, pool)
        for i in range(2):
            for i in range(25):
                cl.collect_new_steps(5,125,step_mode='init')
        batch = pool.random_batch_torch(20)
        print(batch['observations'].shape)
        print(batch['next_observations'].shape)
        print(batch['rewards'])
        print(batch['actions'])
        print(batch['terminals'])
        print(batch['states'].shape)
        print(batch['next_states'].shape)
    