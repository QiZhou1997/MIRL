from typing import Optional, Tuple, Union, List
from collections import OrderedDict
import numpy as np
from collections import deque

from mirl.external_libs.robosuite_wrapper import robosuite_container
from mirl.environments.base_env import Env
from mirl.utils.logger import logger

#TODO: 重构
class RobosuiteEnv(Env):
    def __init__(
        self, 
        distracting='all', #None, "none", "camera", "color", "background", "easy", "medium", "hard"
        dynamic_distracting=False,
        action_repeat: int = 1,
        image_size: int = 168,
        frame_stack: int = 3,
        obs_before_reset: str = "repeat",
        return_state: bool = False,
        return_traj: bool = False,
        record_video: bool = False,
        render_via_env: bool = False,
        env_render_size: int = 480,
        video_fps: int = 10,
        video_dir: Optional[str] = None,
        video_prefix: str = 'dmc_video_',
        **inner_env_kwargs
    ) -> None:
        assert action_repeat == 1
        assert frame_stack == 3
        assert return_traj == False
        assert return_state == False
        assert record_video == False
        #obs_befor_reset: "repeat" or "zero"
        #action and reward: frame_stack - 1 
        self.env_name = "Door"
        self.obs_before_reset = obs_before_reset
        pixels_only = not return_state
        if distracting == "all":
            distracting = ['camera', 'light', 'color']
        if distracting is None or distracting == 'no':
            distracting = []
        env = robosuite_container.make(
            env_name="Door", # ["Door", "TwoArmPegInHole", "NutAssemblyRound", "TwoArmLift"]
            robots="Panda", # ["Panda", "Kinova3", "Jaco", "Sawyer", "IIWA", "UR5e"]
            mode="eval", # ["train", "eval-easy", "eval-hard", "eval-extreme"]
            episode_length=500,
            image_height=image_size,
            image_width=image_size,
            randomize_camera='camera' in distracting,
            randomize_color='light' in distracting,
            randomize_light='color' in distracting,
            randomize_dynamics=False,
            seed=2
        )
        self.action_repeat = 1
        self._env = env
        # for frame_stack
        self._k = self.frame_stack = frame_stack
        self.return_traj = return_traj
        # construct a new obs space and action space
        self.frame_shape = (3, image_size, image_size)
        self.action_shape = env.action_space.shape
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        #return state
        self.return_state = return_state
        self.state_dim_dict = OrderedDict()
        self.state_size = 0
        self.state_shape = (self.state_size,)
        # for video recording
        self.record_video = record_video

    @property
    def n_env(self):
        return 1

    # r: [(1)] or [(1,1)]
    def reset(self):
        self.cur_step_id = 0
        o = self._env.reset()
        return np.array([o])

    def step(self, action):
        self.cur_step_id = self.cur_step_id + 1
        # policy.action_np
        if len(action.shape) > len(self.action_shape): 
            action = action[0]
        #  step
        o, r, d, _= self._env.step(action)
        o = np.array([o])
        r, d = np.array([[r]]), np.array([[d]])
        return o, r, d, {}
        

    def __getattr__(self, name):
        return getattr(self._env, name)

    def get_reward_done_function(self, known=None):
        raise NotImplementedError

if __name__ == "__main__":
    from tqdm import tqdm
    import time
    env = RobosuiteEnv()
    o = env.reset()
    print(o.shape)
    for i in range(500):
        a = env.action_space.sample()
        o,r,d,_ = env.step(a)
        print(d)

"""
env = dmc2gym.make(
        domain="cheetah",
        task="run",
        visualize_reward=False,
        from_pixels=True,
        height=64,
        width=64,
        frame_skip=4,
    )
"""