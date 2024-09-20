import numpy as np

from enum import Enum

from gymnasium import ObservationWrapper
from gymnasium.spaces import Sequence, Box
from gymnasium.core import ObsType, WrapperObsType

from ocatari.core import OCAtari


class OCWrapper(ObservationWrapper):
    def __init__(self, env: OCAtari):
        super().__init__(env)
        self.reference_list = list(dict.fromkeys(env.reference_list))
        obj_types = Enum('ObjectTypes', self.reference_list, start=0)
        self.num_objects = len(obj_types)

        old_shape = env.observation_space.shape
        shape = (old_shape[1], old_shape[0] * old_shape[2] + self.num_objects)
        self.observation_space = Sequence(Box(0, 255, shape=shape),
                                          stack=True)

        ones = [obj_types[t].value for t in env.reference_list]
        self.obs = np.eye(self.num_objects, shape[1])[ones]

    def observation(self, observation: ObsType) -> WrapperObsType:
        new_obs = np.swapaxes(observation, 0, 1)
        self.obs[:, self.num_objects:] = new_obs.reshape(new_obs.shape[0], -1)
        return self.obs