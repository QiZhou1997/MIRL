import abc
import gtimer as gt
from collections import OrderedDict

from mirl.algorithms.off_policy_algorithm import OffPolicyRLAlgorithm
from mirl.utils.eval_util import get_generic_path_information
from mirl.utils.process import Progress, Silent, format_for_process
from mirl.utils.misc_untils import combine_item
from mirl.utils.logger import logger
import mirl.torch_modules.utils as ptu
import numpy as np


class DynaStyleAlgorithm(OffPolicyRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            num_epochs: int = 1000,
            batch_size: int = 256,
            num_train_loops_per_epoch: int = 1000,
            num_expl_steps_per_train_loop: int = 1,
            num_trains_per_train_loop: int = 20,
            num_eval_steps: int = 5000,
            eval_freq: bool = 1,
            max_path_length: int = 1000,
            min_num_steps_before_training: int = 5000,
            silent: bool = False,
            record_video_freq: int = 50,
            analyze_freq: int = -1,
            save_pool_freq: int = -1,
            real_data_ratio: float = 0.1,
            train_model_freq: int = 250,
            imagine_freq: int = 250,
            separate_batch: bool = False,
            item_dict_config: dict = {},
    ):
        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_train_loops_per_epoch=num_train_loops_per_epoch,
            num_expl_steps_per_train_loop=num_expl_steps_per_train_loop,
            num_trains_per_train_loop=num_trains_per_train_loop,
            num_eval_steps=num_eval_steps,
            eval_freq=eval_freq,
            max_path_length=max_path_length,
            min_num_steps_before_training=min_num_steps_before_training,
            silent=silent,
            record_video_freq=record_video_freq,
            analyze_freq=analyze_freq,
            save_pool_freq=save_pool_freq,
            item_dict_config=item_dict_config,
        )
        self.real_data_ratio = real_data_ratio
        self.train_model_freq = train_model_freq
        self.imagine_freq = imagine_freq
        assert (imagine_freq % train_model_freq) == 0
        self._need_snapshot.append('model')
        o_shape = self.expl_env.observation_space.shape
        extra_fields = {
            'deltas': {
                'shape': o_shape,
                'type': np.float32,
            },
        }
        self.pool.add_extra_fields(extra_fields)
        self.real_batch_size = int(batch_size * real_data_ratio)
        self.imagined_batch_size = batch_size - self.real_batch_size
        self.separate_batch = separate_batch
        self.model_diagnostics = {}

    def _train_model(self):
        self.model_diagnostics = self.model_trainer.train_with_pool(self.pool)

    def _train_epoch(self, epoch):
        progress = self.progress_class(self.num_train_loops_per_epoch * self.num_trains_per_train_loop)
        for i in range(self.num_train_loops_per_epoch):
            if i % self.train_model_freq == 0:
                self.training_mode(True)
                self._train_model()
                self.training_mode(False)
                gt.stamp('training model', unique=False)
            if i % self.imagine_freq == 0:
                self.model_collector.imagine(epoch)
                gt.stamp('rollout', unique=False)
            self._sample(self.num_expl_steps_per_train_loop)
            gt.stamp('exploration sampling', unique=False)
            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                progress.update()
                params = self._train_batch()
                params['Model Train Loss'] = self.model_diagnostics['train_loss']
                params['Model Eval Loss'] = self.model_diagnostics['eval_loss']
                progress.set_description(format_for_process(params))
            self.training_mode(False)
            gt.stamp('training', unique=False)
        expl_return = self.expl_collector.get_diagnostics()['Return Mean']
        logger.tb_add_scalar("return/exploration", expl_return, epoch)
        # evaluation
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

    def _train_batch(self):
        real_batch = self.pool.random_batch_torch(self.real_batch_size, without_keys=['deltas'])
        imagined_batch= self.imagined_data_pool.random_batch_torch(self.imagined_batch_size)
        gt.stamp('sample torch batch', unique=False)
        if self.separate_batch:
            params = self.agent.train_from_torch_batch(real_batch, imagined_batch)
        else:
            train_data = combine_item(real_batch, imagined_batch)
            params = self.agent.train_from_torch_batch(train_data)
        return params

    def log_stats(self, epoch):
        logger.record_dict(
            self.model_diagnostics,
            prefix='model/'
        )
        logger.record_dict(
            self.imagined_data_pool.get_diagnostics(),
            prefix='model_pool/'
        )
        super().log_stats(epoch)


