import abc
import gtimer as gt
import os
from mirl.algorithms.base_algorithm import RLAlgorithm
from mirl.utils.eval_util import get_generic_path_information
from mirl.utils.process import Progress, Silent, format_for_process
from mirl.collectors.utils import rollout

from mirl.utils.logger import logger

class OfflineRLAlgorithm(RLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        num_epochs,
        batch_size,
        num_train_loops_per_epoch,
        num_trains_per_train_loop,
        num_eval_steps=5000,
        eval_freq=1,
        max_path_length=1000,
        silent = False,
        record_video_freq=50,
        analyze_freq=1,
        item_dict_config={},
    ):
        super().__init__(item_dict_config)
        self._need_snapshot.append('agent')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_eval_steps = num_eval_steps
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.max_path_length = max_path_length
        self.record_video_freq = record_video_freq
        self.progress_class = Silent if silent else Progress
        self.collected_samples = 0
        self.analyze_freq = analyze_freq
        self.eval_freq = eval_freq
    
    def _end_epoch(self, epoch):
        if (
                self.analyze_freq > 0 and epoch>=0 and \
                    (
                        epoch % self.analyze_freq == 0 or \
                        epoch == self.num_epochs-1
                    ) and \
                hasattr(self, 'analyzer')
            ):
            self.analyzer.analyze(epoch)
        gt.stamp('analyze', unique=False)
        if (
                self.record_video_freq > 0 and epoch>=0 and \
                (
                    epoch % self.record_video_freq == 0 or \
                    epoch == self.num_epochs-1
                ) and \
                hasattr(self, 'video_env')
            ):
            self.video_env.set_video_name("epoch{}".format(epoch))
            logger.log("rollout to save video...")
            self.video_env.recored_video(
                self.agent, 
                max_path_length=self.max_path_length, 
                use_tqdm=True,
                step_mode='exploit'
            )
        gt.stamp('save video', unique=False)

    def _before_train(self):
        self.training_mode(True)
        self.agent.pretrain(self.pool)
        self.training_mode(False)

    def _train_epoch(self, epoch):
        progress = self.progress_class(self.num_train_loops_per_epoch * self.num_trains_per_train_loop)
        for _ in range(self.num_train_loops_per_epoch):
            # trainning
            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                progress.update()
                # train_data = self.pool.random_batch(self.batch_size)
                # gt.stamp('sample from pool', unique=False)
                # batch = ptu.np_to_pytorch_batch(train_data)
                # gt.stamp('cpu to gpu', unique=False)
                torch_batch = self.pool.random_batch_torch(self.batch_size)
                gt.stamp('sample torch batch', unique=False)
                params = self.agent.train_from_torch_batch(torch_batch)
                progress.set_description(format_for_process(params))
                gt.stamp('training', unique=False)
            self.training_mode(False)

        if (
            self.eval_freq > 0 and epoch>=0 and (
                epoch % self.eval_freq == 0 or \
                epoch == self.num_epochs-1
            ) and hasattr(self, 'eval_collector')
        ):
            self.eval_collector.collect_new_steps(
                self.num_eval_steps,
                self.max_path_length,
                step_mode='exploit'
            )
            eval_return = self.eval_collector.get_diagnostics()['Return Mean']
            logger.tb_add_scalar("return/evaluation", eval_return, epoch)
        logger.tb_flush()
        gt.stamp('evaluation sampling')
        progress.close()