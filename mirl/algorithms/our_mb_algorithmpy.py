from mirl.algorithms.dyna_algorithm import DynaStyleAlgorithm

class OurMBAlgorithm(DynaStyleAlgorithm):
    def _train_batch(self):
        real_batch = self.pool.random_batch(self.real_batch_size, without_keys=['deltas'])
        imagined_batch= self.imagined_data_pool.random_batch(self.imagined_batch_size)
        params = self.agent.train(real_batch, imagined_batch)
        return params


