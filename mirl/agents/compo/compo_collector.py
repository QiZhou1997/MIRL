from mirl.utils.misc_untils import get_scheduled_value
from mirl.agents.mbpo.mopo_collector import MBPOCollector
import mirl.torch_modules.utils as ptu
import torch

class COMPOCollector(MBPOCollector):
    def __init__(
        self, 
        model, 
        policy, 
        pool,
        imagined_data_pool, 
        random_sample=False,
        **kwargs
    ):
        super().__init__(
            model=model,
            policy=policy,
            pool=pool,
            imagined_data_pool=imagined_data_pool,
            **kwargs
        )
        self.random_sample = random_sample

    def imagine(self, cur_epoch):
        states = self.get_start_states()
        depth = self.cur_depth = self.get_depth(cur_epoch)
        self.resize_pool(cur_epoch)
        batch_size = self.batch_size
        number_sample = self.number_sample
        if batch_size is None:
            batch_size = number_sample
        with torch.no_grad():
            for i in range(0, number_sample, batch_size):
                cur_size = min(batch_size, number_sample-i)
                o=states[i:i+cur_size]
                stop = ptu.zeros((cur_size,1))
                for j in range(depth):
                    a,_ = self.policy.action(o)
                    if self.random_sample:
                        a = torch.rand_like(a)*2-1
                    next_o, r, d, info = self.model.step(o, a, **self.step_kwargs)
                    d = stop+d - stop*d # stop or d
                    o, stop = self.add_sample(stop,o,a,next_o,r,d,info,j,cur_epoch)
                    if len(stop)==0 or torch.sum(stop)==len(stop):
                        break
        self._n_image_times_total += 1