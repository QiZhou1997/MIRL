import numpy as np
import warnings
from collections import OrderedDict

from mirl.pools.simple_pool import SimplePool
import copy

class FlagPool(SimplePool):
    def __init__(self, env, max_size=1e6, compute_mean_std=False):
        super().__init__(env, max_size, compute_mean_std)
        self.unprocessed_size={}
    
    def add_samples(self, samples):
        new_stop = None
        for k in self.fields:
            v = samples[k]
            if new_stop is None:
                new_sample_size, new_stop = self._update_single_field(k,v)
            else:
                _, _new_stop = self._update_single_field(k,v)
                assert _new_stop == new_stop
        self._stop = new_stop
        self._size = min(self.max_size, self._size + new_sample_size)
        for tag in self.unprocessed_size:
            unprocessed_size = self.unprocessed_size[tag] + new_sample_size
            if unprocessed_size > self.max_size:
                warnings.warn("unprocessed_size > max_size")
                self.unprocessed_size[tag] = self.max_size
            else:
                self.unprocessed_size[tag] = unprocessed_size
        return new_sample_size
    
    def get_unprocessed_data(self, tag, keys=None, without_keys=[]):
        keys = self._check_keys(keys, without_keys)
        if tag not in self.unprocessed_size:
            self.unprocessed_size[tag] = self._size
        stop = self._stop
        size = self.unprocessed_size[tag]
        data = {}
        for key in keys:
            assert key in self.fields
            temp_data = self.dataset[key]
            if size > stop:
                data[key] = np.concatenate((temp_data[stop-size:], temp_data[:stop]))
            else:
                data[key] = temp_data[stop-size:stop]
        return data

    def update_process_flag(self, tag, process_num):
        if tag not in self.unprocessed_size:
            self.unprocessed_size[tag] = self._size
        assert process_num <= self.unprocessed_size[tag]
        self.unprocessed_size[tag] -= process_num
    
    def resize(self, size, **init_kwargs):
        # TODO：很慢
        from mirl.utils.misc_untils import dotdict
        assert size > self._size
        samples = self.get_data(self.sample_keys)
        unprocessed_size = copy.deepcopy(self.unprocessed_size)
        for tag in unprocessed_size:
            assert unprocessed_size[tag] == 0
        env = dotdict({
            'action_space': dotdict({'shape': self._action_shape}),
            'observation_space': dotdict({'shape': self._observation_shape}),
        })
        self.__init__(env, size, compute_mean_std=self.compute_mean_std, **init_kwargs)
        self.add_samples(samples)
        self.unprocessed_size = unprocessed_size
        