import numpy as np
import gymnasium as gym

from gymnasium.envs.toy_text.taxi import TaxiEnv


class OCTaxi(gym.ObservationWrapper):
    metadata = TaxiEnv.metadata

    def __init__(self, **kwargs):
        super().__init__(env=gym.make("Taxi-v3",  **kwargs))
        self.observation_space = gym.spaces.Box(0, 16,(3, 5))
        self.first = True

    def reset(self, *, seed=None, options=None):
        self.first = True
        return super().reset(seed=seed, options=options)

    def step(self, action):
        tup = list(super().step(action))

        if self.first and action == 4 and tup[1] == -1:
            self.first = False
            tup[1] = 10

        return tuple(tup)

    def observation(self, observation):
        new_obs = np.zeros((3, 5), dtype=np.float32)

        # destination
        dest = observation % 4
        observation -= dest
        observation /= 4
        new_obs[0 , 0] = 1
        if dest == 0:
            new_obs [0, 3:5] = (0, 0)
        elif dest == 1:
            new_obs [0, 3:5] = (0, 4)
        elif dest == 2:
            new_obs[0, 3:5] = (4, 0)
        elif dest == 3:
            new_obs[0, 3:5] = (4, 3)

        # passenger
        pas = observation % 5
        observation -= pas
        observation /= 5
        if pas == 0:
            new_obs[1, 3:5] = (0, 0)
            new_obs[1, 1] = 1
        elif pas == 1:
            new_obs[1, 3:5] = (0, 4)
            new_obs[1, 1] = 1
        elif pas == 2:
            new_obs[1, 3:5] = (4, 0)
            new_obs[1, 1] = 1
        elif pas == 3:
            new_obs[1, 3:5] = (4, 3)
            new_obs[1, 1] = 1

        # player
        new_obs[2 , 2] = 1
        new_obs[2, 3] = observation // 5
        new_obs[2, 4] = observation % 5

        return new_obs
