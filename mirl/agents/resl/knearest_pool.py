from matplotlib.pyplot import get
import numpy as np
from mirl.pools.simple_pool import SimplePool
import gym
import faiss

class KNearestOfflinePool(SimplePool):
    def __init__(
        self, 
        dataset_name, 
        k=10,
    ):
        import d4rl
        env = gym.make(dataset_name)
        d = self.origin_dataset = d4rl.qlearning_dataset(env)
        size = len(d['rewards'])
        super().__init__(env, max_size=size, compute_mean_std=True)
        d['rewards'] = d['rewards'][:,None]
        d['terminals'] = d['terminals'][:,None]
        self.add_samples(d)
        obs_std = self.dataset_mean_std['observations'].std+1e-6
        self.obs_std = obs_std.astype(np.float32)
        norm_obs = d['observations'] / self.obs_std
        norm_next_obs = d['next_observations'] / self.obs_std
        obs_dim = norm_obs.shape[-1]
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(obs_dim)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index_flat.add(norm_obs)    
        self.index_flat = gpu_index_flat
        self.k = k
        print("search k neartest for observations")
        self.distance, self.ind = gpu_index_flat.search(norm_obs, k)  # 
        print("search k neartest for next observations")
        self.next_distance, self.next_ind = gpu_index_flat.search(norm_next_obs, k)  # 
        
    def search(self, obs, k=None, norm=True):
        if k is None:
            k = self.k
        if norm:
            obs = obs/self.obs_std
        batch_size = len(obs)
        distance, ind = self.index_flat.search(obs.astype(np.float32), k)
        k_actions = self.dataset['actions'][ind]
        k_actions = k_actions.reshape((batch_size,k,-1))
        return k_actions, distance

    def _update_batch(self, batch_index, batch, prefix=''):
        batch_size = len(batch_index)
        ind = getattr(self, prefix+'ind')
        ind = ind[batch_index]
        ind = ind.reshape((batch_size*self.k,))
        k_actions = self.dataset['actions'][ind]
        k_actions = k_actions.reshape((batch_size,self.k,-1))
        distance = getattr(self, prefix+'distance')
        obs_distance = distance[batch_index]
        batch[prefix+'k_actions'] = k_actions
        batch[prefix+'obs_distance'] = obs_distance
    
    def random_batch(self, batch_size, keys=None, without_keys=[]):
        assert keys is None and len(without_keys) == 0
        batch_index = np.random.randint(0, self._size, batch_size)
        batch = {}
        for key in self.fields:
            value = self.dataset[key]
            batch[key] = value[batch_index]
        self._update_batch(batch_index, batch)
        self._update_batch(batch_index, batch, 'next_')
        return batch

if __name__ == '__main__':
    from mirl.environments.video_env import VideoEnv
    pool = RESLOfflinePool('hopper-expert-v2')
    print(pool.distance.mean())
    print(pool.obs_std)
    env = VideoEnv('Hopper-v3', directory='/home/qzhou/videos')
    o = env.reset()
    total_r = 0
    a, d = pool.search(o.astype(np.float32), k=1)
    done = False
    for i in range(1000):
        if done:
            break
        a = a.mean(1)
        o, r, done, _ = env.step(a)
        a, d = pool.search(o.astype(np.float32), k=1)
        total_r += r
        print(r, d.mean(), done)
    print(total_r)

