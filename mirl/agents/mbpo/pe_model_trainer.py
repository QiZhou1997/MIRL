from collections import OrderedDict
from genericpath import exists

import numpy as np
import torch.optim as optim
import torch
import os
import os.path as osp
from os.path import join

import mirl.torch_modules.utils as ptu
from mirl.components import Component
from mirl.pools.utils import split_dataset
from mirl.utils.process import Progress, Silent, format_for_process
from mirl.utils.logger import logger


class PEModelTrainer(Component):
    def __init__(
        self,
        env,
        model,
        lr=1e-3,
        max_step_per_train=4000,
        init_model_train_step=int(5e4),
        load_best=True,
        valid_ratio=0.2,
        max_valid=2000,   
        resample=True,             
        batch_size=256, 
        report_freq=100,
        max_not_improve=20,
        silent=False,
        load_dataset_dir=None,
        tb_log_freq=100,
        optimizer_class='Adam',
        optimizer_kwargs={},
        load_dir=None,
        save_dir=None,
        ignore_terminal_state=False,
    ):
        if isinstance(optimizer_class, str):
            optimizer_class = eval('optim.'+optimizer_class)
        self.env = env
        self.model = model
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=lr,
            **self.optimizer_kwargs
        )
        self.statistics = OrderedDict()
        self.learn_reward = model.learn_reward
        self.learn_done = model.learn_done
        self.num_train_steps = 0
        self.max_step_per_train = max_step_per_train
        self.init_model_train_step = int(init_model_train_step)
        self.load_best = load_best
        self.tb_log_freq = tb_log_freq
        self.ignore_terminal_state = ignore_terminal_state
        self.valid_ratio = valid_ratio
        self.max_valid = max_valid
        self.resample = resample   
        self.batch_size = batch_size 
        self.report_freq = report_freq
        self.max_not_improve = max_not_improve
        self.silent = silent
        self.load_dataset_dir = load_dataset_dir
        if load_dir is not None:
            self.load_dir = osp.expanduser(load_dir)
        else:
            self.load_dir = None
        if save_dir is not None:
            self.save_dir = osp.expanduser(save_dir)
        else:
            self.save_dir = None
        
    def _log_tb_or_not(self):
        if self.tb_log_freq > 0 and logger.log_or_not(logger.ERROR) \
            and (self.num_train_steps % self.tb_log_freq==0):
            return True
        else:
            return False

    def train_from_torch_batch(self, batch):
        o = batch['observations']
        a = batch['actions']
        deltas = batch['deltas']
        r = batch['rewards']
        d = batch['terminals']
        loss, ensemble_loss = self.model.compute_loss(
            o,a,deltas,r,d,
            self.ignore_terminal_state, 
            self.num_train_steps,
            self._log_tb_or_not()
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        diagnostics = OrderedDict()
        ensembel_loss_np = ptu.get_numpy(ensemble_loss)
        for i,l in enumerate(ensembel_loss_np):
            diagnostics['mt/net_%d/train_loss'%i] = l
        diagnostics['train_loss'] = loss.item()
        self.num_train_steps += 1
        return diagnostics

    def eval_with_dataset(self, dataset):
        dataset = ptu.np_to_pytorch_batch(dataset)
        o = dataset['observations']
        a = dataset['actions']
        deltas = dataset['deltas']
        r = dataset['rewards']
        d = dataset['terminals']
        loss, ensemble_loss= self.model.compute_loss(o,a,deltas,r,d)
        diagnostics = OrderedDict()
        ensemble_loss_np = ptu.get_numpy(ensemble_loss)
        for i,l in enumerate(ensemble_loss_np):
            diagnostics['mt/net_%d/eval_loss'%i] = l
        diagnostics['eval_loss'] = loss.item()
        return diagnostics, list(ensemble_loss_np)

    def set_mean_std(self, mean_std_dict=None):
        model = self.model
        if mean_std_dict is None:
            mean_std_dict = self.mean_std_dict
        else:
            self.mean_std_dict = mean_std_dict
        if model.normalize_obs:
            model.obs_processor.set_mean_std_np(*mean_std_dict['observations'])
        if model.normalize_action:
            model.action_processor.set_mean_std_np(*mean_std_dict['actions'])
        if model.normalize_delta:
            model.delta_processor.set_mean_std_np(*mean_std_dict['deltas'])
        if model.normalize_reward:
            model.reward_processor.set_mean_std_np(*mean_std_dict['rewards'])

    def train_with_pool(self, pool):       
        max_step = self.max_step_per_train 
        temp_data = pool.get_unprocessed_data('compute_delta', ['observations', 'next_observations'])
        deltas = temp_data['next_observations'] - temp_data['observations']
        pool.update_single_extra_field('deltas', deltas)
        pool.update_process_flag('compute_delta', len(deltas))
        self.set_mean_std(pool.get_mean_std())
        if self.num_train_steps == 0:
            self.init_model = False
            max_step = self.init_model_train_step
        else:
            max_step = int(max_step)
        if self.load_dir is not None:
            succes = self.model.load(self.load_dir)
            if not succes:
                logger.log("Can not load a model from %s"%self.load_dir)
        resutls = self.sufficiently_train(
            pool, 
            max_step, 
            valid_ratio=self.valid_ratio,
            max_valid=self.max_valid,   
            resample=self.resample,             
            batch_size=self.batch_size, 
            report_freq=self.report_freq,
            max_not_improve=self.max_not_improve,
            load_best = self.load_best,
            silent=self.silent,
            load_dataset_dir=self.load_dataset_dir
        )
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            self.model.save(self.save_dir)
        return resutls

    def split_dataset(self, pool, valid_ratio, max_valid, resample):
        ensemble_size = self.model.ensemble_size
        dataset = pool.get_data()
        train_dataset, eval_dataset, train_length, eval_length = split_dataset(dataset, valid_ratio, max_valid)
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.train_length, self.eval_length = train_length, eval_length
        if resample:
            self.ensemble_index = np.random.randint(train_length, size=(ensemble_size, train_length))
        else:
            ensemble_index = np.arange(train_length)[None]
            self.ensemble_index = np.tile(ensemble_index, (ensemble_size,1))

    def save_dataset(self, save_dir=None):
        save_dic = {
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
            "train_length": self.train_length,
            "eval_length": self.eval_length,
            "ensemble_index": self.ensemble_index,
            "mean_std_dict": self.mean_std_dict
        }
        if save_dir == None:
            save_dir = logger._snapshot_dir
        path = join(save_dir, 'pe_model_dataset.npy')
        np.save(path, save_dic, allow_pickle=True)
    
    
    def load_dataset(self, load_dir=None):
        if load_dir == None:
            load_dir = logger._snapshot_dir
        path = join(load_dir, 'pe_model_dataset.npy')
        load_dic = np.load(path, allow_pickle=True).item()
        self.__dict__.update(load_dic)
    
    def _get_batch_from_dataset(self, dataset, index):
        batch = {}
        for k in dataset:
            batch[k] = dataset[k][index]
        return batch
    
    def sufficiently_train(
        self,
        pool=None,
        max_step=2000, 
        valid_ratio=0.2,
        max_valid=5000,   
        resample=True,             
        batch_size=256, 
        report_freq=100,
        max_not_improve=20,
        load_best=True,
        silent=False,
        load_dataset_dir=None
    ):
        ### test_code
        model = self.model
        if pool is not None:
            self.split_dataset(pool, valid_ratio, max_valid, resample)
        else:
            assert load_dataset_dir is not None
            self.load_dataset(load_dataset_dir)
            self.set_mean_std()
        train_dataset, eval_dataset = self.train_dataset, self.eval_dataset
        train_length = self.train_length
        ensemble_index = self.ensemble_index
        progress_class = Silent if silent else Progress
        progress = progress_class(int(max_step))
        total_train_step = 0
        while True:
            for j in range(0,train_length,batch_size):
                progress.update()
                if total_train_step % report_freq == 0:
                    eval_stat, eval_loss = self.eval_with_dataset(eval_dataset)
                    if total_train_step == 0:
                        min_loss = eval_loss
                        not_improve = []
                        for i,_ in enumerate(eval_loss):
                            model.save(net_id=i)
                            not_improve.append(0) 
                    else:
                        for i,l in enumerate(eval_loss):
                            if l < min_loss[i]:
                                min_loss[i] = l
                                not_improve[i] = 0
                                model.save(net_id=i)
                            else:
                                not_improve[i] += 1
                    continue_training = False
                    for i,n in enumerate(not_improve):
                        eval_stat['mt/net_%d/not_improve'%i] = n
                        if max_not_improve<0 or n<max_not_improve:
                            continue_training = True
                if total_train_step >= max_step:
                    continue_training = False
                if not continue_training:
                    break
                ind = ensemble_index[:, j:j+batch_size]
                train_batch = self._get_batch_from_dataset(train_dataset, ind)
                train_batch = ptu.np_to_pytorch_batch(train_batch)
                train_stat = self.train_from_torch_batch(train_batch)
                stat = OrderedDict( list(train_stat.items()) + list(eval_stat.items()) )
                progress.set_description(format_for_process(stat))
                total_train_step += 1
                if self._log_tb_or_not():
                    for k in stat:
                        if "not_improve" in k:
                            continue
                        logger.tb_add_scalar(k, stat[k], self.num_train_steps)
            if not continue_training:
                break
        if load_best:
            for i in range(len(eval_loss)):
                model.load(net_id=i)
        #load时不会load log_std_min 和 log_std_max
        progress.close()
        eval_stat, eval_loss = self.eval_with_dataset(eval_dataset)
        model.remember_loss(eval_loss)
        self.statistics.update(train_stat)
        self.statistics.update(eval_stat)
        self.statistics['train_step'] = total_train_step
        self.min_loss = eval_loss
        return self.statistics

    def get_diagnostics(self):
        return self.statistics

    def end_epoch(self, epoch):
        pass

    @property
    def networks(self):
        return [self.model]

    def get_snapshot(self):
        return dict(model=self.model)


if __name__ == "__main__":
    from mirl.pools.offline_pool import OfflinePool
    from mirl.agents.mbpo.pe_model import PEModel
    from mirl.environments.base_env import SimpleEnv
    seed = np.random.randint(88888)
    logger.set_snapshot_dir("/home/qizhou/data/modelbased/%d"%seed)
    import gym
    pool = OfflinePool(
        "hopper-medium-v2", 
        "ExtraFieldPool",
        compute_mean_std=True
    )
    kwargs = {
        "hidden_layers": [200,200,200,200],
        "activation": "swish",
        "connection": "densenet",
        "reward_coef": 1,
        "bound_reg_coef": 2e-2,
        "weight_decay": [2.5e-5,5e-5,7.5e-5,7.5e-5,1e-4]
    }
    env =  SimpleEnv("Hopper-v2")
    model = PEModel(
        env,
        known=['done'],
        **kwargs,
    )
    ptu.set_gpu_mode(True)
    model.to(ptu.device)
    logger.set_log_level('INFO')
    o_shape = env.observation_space.shape
    extra_fields = {
        'deltas': {
            'shape': o_shape,
            'type': np.float32,
        },
    }
    pool.add_extra_fields(extra_fields)
    trainer = PEModelTrainer(
        env, model,
        ignore_terminal_state=True,
        init_model_train_step=50000
    )
    trainer.train_with_pool(pool)
    batch = pool.random_batch_torch(666)
    pred_next_obs, r, done, _ = model.step(batch['observations'], batch['actions'])
    print(r-batch['rewards'], batch['rewards'])
    print( (done-batch['terminals']).abs().sum(), batch['terminals'].sum())
    