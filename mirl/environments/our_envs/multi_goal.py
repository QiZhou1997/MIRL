
import numpy as np

import gym
import matplotlib.pyplot as plt
from gym import spaces
from gym.utils import seeding

GOALS = np.array([[15,0]])
# GOALS = np.array([[0,5], [0,-5], [5,0], [-5,0]])
#GOALS = np.array([[4,4], [-4,-4], [4,-4], [-4,4]])
TEMPERATURE = 50
SCALE = 10
BOUND = np.array([20,20])

class MultiGoal2DEnv(gym.Env):
    def __init__(self, random_reset=True, goals=GOALS, bound=BOUND, temperature=TEMPERATURE):
        self.bound = np.array(bound)
        self.random_reset = random_reset
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low = -max(self.bound),
            high = max(self.bound),
            shape=(2,),
            dtype=np.float32
        )
        self.goals = np.array(goals)
        self.temperature = temperature
        self.seed()
        self.sim = self

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        r = self._reward(self.state, action)
        s = self.state + np.clip(action, -1, 1)
        s = self.state = np.clip(s, -self.bound, self.bound)
        return s, r, False, {}
    
    def _reward(self, s, a):
        next_s = s + np.clip(a, -1, 1)
        r = 0
        for g in self.goals:
            r += np.exp(-np.square(next_s-g).sum()/self.temperature)*SCALE
        return r

    def reset(self):
        if self.random_reset:
            self.state = np.random.rand(2,)*2*self.bound-self.bound
        else:
            self.state = np.array([0,0])
        return self.state

    def get_state(self):
        return self.state
    
    def set_state(self, state):
        self.state = state

    def state_vector(self):
        return self.state

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

def f_reward(X,Y,goals=GOALS,temperature=TEMPERATURE):
    S = np.concatenate([X.reshape(X.shape+(1,)), Y.reshape(Y.shape+(1,))], axis=-1)
    r = 0
    for g in goals:
        r += np.exp(-np.square(S-g).sum(-1)/temperature)*SCALE
    return r

def ftos(v):
    s = '%.1f'%v
    return '%.0f'%v if s[-1] == '0' else s

if __name__ == "__main__":
    n = 256
    bound = BOUND+0.5
    x = np.linspace(-bound, bound, n)
    y = np.linspace(-bound, bound, n)
    X,Y = np.meshgrid(x, y)

    # Basic contour plot
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, f_reward(X,Y), levels = 12, linewidths = 1)

    # Recast levels to new class
    CS.levels = [ftos(val) for val in CS.levels]

    ax.set_aspect('equal')
    ax.clabel(CS, CS.levels, inline=True, fontsize=6) # , CS.levels
    plt.show()
