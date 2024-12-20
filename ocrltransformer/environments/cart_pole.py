import numpy as np
import gymnasium as gym

from gymnasium.envs.classic_control import CartPoleEnv


class OCCartPole(gym.ObservationWrapper):
    metadata = CartPoleEnv.metadata

    def __init__(self, **kwargs):
        super().__init__(env=gym.make("CartPole-v1",  **kwargs))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,(2, 3))

    def observation(self, observation):
        new_obs = np.zeros((2, 3))
        new_obs[..., 1:] = observation.reshape(2, 2)
        new_obs[0, 0] = 1
        return new_obs