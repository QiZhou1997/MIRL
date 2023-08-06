import mirl.torch_modules.utils as ptu

#TODO 添加checkpoint
class Pool(object):
    def __init__(self, env):
        pass

    def add_samples(self, samples):
        raise NotImplementedError

    def add_paths(self, paths):
        raise NotImplementedError
    
    def random_batch(self, batch_size):
        pass

    def random_batch_torch(self, batch_size, **kwargs):
        batch = self.random_batch(batch_size, **kwargs)
        return ptu.np_to_pytorch_batch(batch)

    def get_diagnostics(self):
        return {}

    def start_new_path(self, o):
        pass

    def end_a_path(self, next_o):
        pass