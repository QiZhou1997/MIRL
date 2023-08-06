# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# from drqv2
from collections import deque
from random import randint
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_control.suite import common
from dm_env import StepType, specs
import mirl.external_libs.distracting_control.suite as dcs_suite
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
import os
import copy

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


N = 3
class _ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats
        self.rew_deque = deque(maxlen=N)
        for i in range(N):
            self.rew_deque.append([0,-1])

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            nstep = np.random.randint(N)
            self.rew_deque.append([reward, nstep])
            reward = 0
            for pair in self.rew_deque:
                if pair[1] == 0:
                    reward += pair[0]
                pair[1] = pair[1] - 1
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

p = 0.25
class __ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            if np.random.rand() > p:
                self.obs = time_step.observation
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break
        return time_step._replace(reward=reward, discount=discount, observation=self.obs)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        time_step = self._env.reset()
        self.obs = time_step.observation
        return time_step

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class SimpleFramekWrapper(dm_env.Environment):
    def __init__(self, env, pixels_key='pixels'):
        self._env = env
        self._pixels_key = pixels_key
        inner_obs_spec = env.observation_spec()
        assert pixels_key in inner_obs_spec
        pixels_shape = inner_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._pixel_spec = specs.BoundedArray(shape=np.array(
            [pixels_shape[2], *pixels_shape[:2]]),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )
        inner_obs_spec[pixels_key] = self._pixel_spec
        self._obs_spec = inner_obs_spec
        
    def _trans_pixels(self, time_step):
        obs = copy.deepcopy(time_step.observation)
        pixels = obs[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        obs[self._pixels_key]=pixels.transpose(2, 0, 1).copy()
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        time_step = self._trans_pixels(time_step)
        return time_step

    def step(self, action):
        time_step = self._env.step(action)
        time_step = self._trans_pixels(time_step)
        return time_step

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

def _make(
    domain, 
    task,
    action_repeat=1, 
    height=84,
    width=84,
    pixels_only=True,
    **kwargs
):
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(
            domain,
            task,
            visualize_reward=False,
            **kwargs
        )
        pixels_key = 'pixels'
        is_manipulation_task = False
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, **kwargs)
        pixels_key = 'front_close'
        is_manipulation_task = True
        raise NotImplementedError
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(
            height=height, 
            width=width, 
            camera_id=camera_id
        )
        env = pixels.Wrapper(env,
                             pixels_only=pixels_only,
                             render_kwargs=render_kwargs)
    env = SimpleFramekWrapper(env)
    env = ExtendedTimeStepWrapper(env)
    return env


DEFAULT_BACKGROUND_PATH = os.path.expanduser('~/datasets/DAVIS/JPEGImages/480p/')
def _make_dcs(
    domain, 
    task,
    dataset_path=DEFAULT_BACKGROUND_PATH,
    dataset_videos='train',
    action_repeat=1, 
    height=84,
    width=84,
    pixels_only=True,
    **kwargs
):
    assert (domain, task) in suite.ALL_TASKS, "%s, %s"%(domain, task)
    render_kwargs = dict(height=height, width=width)
    env = dcs_suite.load(
        domain,
        task,
        background_dataset_path=dataset_path,
        background_dataset_videos=dataset_videos,
        pixels_only=pixels_only,
        render_kwargs=render_kwargs,
        **kwargs
    )
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = SimpleFramekWrapper(env)
    env = ExtendedTimeStepWrapper(env)
    return env


def make(
    domain, 
    task,
    distracting=True,
    **kwargs
):
    if distracting:
        return _make_dcs(domain, task, **kwargs)
    else:
        return _make(domain, task, **kwargs)
