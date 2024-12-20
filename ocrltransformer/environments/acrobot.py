import numpy as np
import gymnasium as gym

from gymnasium.envs.classic_control import AcrobotEnv


class OCAcrobot(gym.ObservationWrapper):
    metadata = AcrobotEnv.metadata

    def __init__(self, **kwargs):
        super().__init__(env=gym.make("Acrobot-v1",  **kwargs))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,(2, 4))

    def observation(self, observation):
        new_obs = np.zeros((2, 4), dtype=np.float32)
        new_obs[..., 1:3] = observation[:4].reshape(2, 2)
        new_obs[..., 3:] = observation[4:].reshape(2, 1)
        new_obs[0, 0] = 1
        return new_obs