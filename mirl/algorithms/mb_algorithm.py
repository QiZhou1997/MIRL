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
import torch
import copy
import pdb


class ModelBasedAlgorithm(OffPolicyRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        train_model_freq=1,
        train_model_kwargs={},
        **offpolicy_algorithm_kwargs
    ):
        super().__init__(**offpolicy_algorithm_kwargs)
        self.train_model_config = train_model_kwargs
        self.train_model_freq = train_model_freq

    def _train_model(self):
        self.model_diagnostics = self.agent.train_model(self.pool, **self.train_model_config)

    def _train_epoch(self, epoch):
        if epoch % self.train_model_freq == 0:
            self._train_model()
        super()._train_epoch(epoch)


