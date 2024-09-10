import numpy as np

from enum import Enum

from gymnasium import ObservationWrapper
from gymnasium.spaces import Sequence, Box
from gymnasium.core import ObsType, WrapperObsType

from ocatari.core import OCAtari


class OCWrapper(ObservationWrapper):
    def __init__(self, env: OCAtari):
        super().__init__(env)
        obj_types = Enum('ObjectTypes', list(set(env.reference_list)), start=0)
        ones = [obj_types[t].value for t in env.reference_list]
        self.types = np.eye(len(obj_types))[ones]

        old_shape = env.observation_space.shape
        shape = (old_shape[1], old_shape[0] * old_shape[2] + len(obj_types))
        self.observation_space = Sequence(Box(0, 255, shape=shape),
                                          stack=True)

    def observation(self, observation: ObsType) -> WrapperObsType:
        new_obs = np.swapaxes(observation, 0, 1)
        new_obs = new_obs.reshape(new_obs.shape[0], -1)
        # new_obs = new_obs[new_obs.sum(axis=1) != 0, :]  # too slow
        new_obs = np.concatenate((self.types, new_obs), axis=1)
        return new_obs