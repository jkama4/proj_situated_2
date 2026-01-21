import gymnasium as gym
import gymnasium_robotics
import numpy as np

from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor


class Float32Wrapper(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space = spaces.Dict({
            k: spaces.Box(
                low=v.low.astype(np.float32),
                high=v.high.astype(np.float32),
                shape=v.shape,
                dtype=np.float32
            )
            for k, v in env.observation_space.spaces.items()
        })
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_obs(obs), reward, terminated, truncated, info
    
    def _convert_obs(self, obs):
        return {k: v.astype(np.float32) for k, v in obs.items()}


def make_env():
    gym.register_envs(gymnasium_robotics)
    env = Monitor(gym.make("FetchPushDense-v4", render_mode="human"))
    return Float32Wrapper(env)