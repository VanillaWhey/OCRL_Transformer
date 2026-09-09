from collections import deque

import numpy as np

import gymnasium as gym
import torch
from gymnasium import ObservationWrapper
from gymnasium.spaces import Sequence, Box

from ocatari.ram.extract_ram_info import get_class_dict, get_max_objects
from ocatari.ram.game_objects import NoObject, GameObject
from torchvision.transforms import RandomCrop, functional as F

from copy import copy

import cv2


class SeqSpace(Sequence):
    @property
    def shape(self):
        return self.feature_space.shape


class EgoCentricWrapper(ObservationWrapper):
    def __init__(self, env, player_name="Player",
                 type_embedding="one_hot", include_wh=False,
                 normalize=False, offset=(0, 0), exclude_classes=()):
        super().__init__(env)

        self.player_name = player_name
        self.relative_pv_index = 4

        if hasattr(env, "max_objs"):
            max_objs = env.max_objs
        else:
            max_objs = get_max_objects(env.game_name, env.hud) # noqa: type(env) == OCAtari
        self.num_object_types = len(max_objs)
        self.max_len = sum(max_objs.values())

        if include_wh:
            self.feature_func = w_h_dx_dy_center
        else:
            self.feature_func = dx_dy_center

        if normalize:
            self.x, self.y = self.unwrapped.ale.getScreenGrayscale().shape
        else:
            self.x, self.y = 1, 1

        self.offset = offset
        self.exclude = exclude_classes

        self.netto_feature_size = len(self.feature_func(NoObject(), self.x, self.y))  # object features only
        self.brutto_feature_size = self.netto_feature_size  # may include type embedding etc.

        if type_embedding == "one_hot":
            self.brutto_feature_size += self.num_object_types
            self.object_types =  {k: np.eye(1, self.brutto_feature_size, i) for i, k in enumerate(max_objs.keys())}
        elif type_embedding is None:
            self.object_types = {k: np.zeros((1, self.brutto_feature_size)) for k in max_objs.keys()}
        else:
            raise AttributeError(f"Type embedding {type_embedding} not supported!")

        self.observation_space = Box(-np.inf, np.inf, shape=(self.max_len, self.brutto_feature_size))


    def observation(self, observation):
        state = np.zeros((self.max_len, self.brutto_feature_size))
        emb = np.zeros((len(self.env.objects), self.brutto_feature_size))
        i = 0
        player_idx = -1
        for o in self.env.objects:  # noqa: type(env) == OCAtari
            if not (o is None or "NoObject" in o.category):
                center_x, center_y = o.center  # only calculate once
                if o.category == self.player_name and player_idx == -1:
                    player_pos_v = [
                        o.dx / self.x,
                        o.dy / self.y,
                        center_x / self.x,
                        center_y / self.y
                    ]
                    player_idx = i
                if o.category in self.exclude:
                    continue
                emb[i] = self.object_types[o.category]
                state[i, -self.netto_feature_size:] = self.feature_func(o, self.x, self.y)
                state[i, -2:] -= self.offset
                i += 1
        if player_idx != -1:  # sometimes the player disappears on termination
            # calculate relative pos and velocity
            state[:i, -self.relative_pv_index:] -= player_pos_v[-self.relative_pv_index:]  # noqa: is always set

        # object type
        state[:i] += emb[:i]

        return state


class ObjFlatObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(-np.inf, np.inf, shape=(env.max_len * env.brutto_feature_size,))

    def observation(self, observation):
        return observation.flatten()


class RandomCropWrapper(RandomCrop, ObservationWrapper):
    def __init__(self, env, padding, obs_mode="ori"):
        super().__init__(env.observation_space.shape[:2], padding, padding_mode="edge")
        super(ObservationWrapper, self).__init__(env)

        self.objects = []
        self.offset_x, self.offset_y = 0, 0
        self.h, self.w = 0, 0

        self._state_buffer_dqn = deque([], 4)

        if type(padding) == int:
            self.pads = (padding, padding)
        else:
            self.pads = padding[:2]

        if obs_mode == "ori":
            self.obs = lambda o: o
        elif obs_mode == "dqn":
            self.obs = self.down_scale
            self._observation_space = gym.spaces.Box(0, 255, (4, 84, 84))
        else:
            raise NotImplementedError(f"obs_mode {obs_mode} is not supported!")

    def down_scale(self, obs):
        dqn_obs = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)
        self._state_buffer_dqn.append(dqn_obs)

        return np.array(self._state_buffer_dqn)

    def get_params(self, img, output_size):
        return self.offset_y, self.offset_x, self.h, self.w

    def observation(self, observation):
        new_obs = self(torch.tensor(observation).moveaxis(-1, 0))

        self.objects = []
        for obj in self.env.objects:
            if not (obj is None or "NoObject" in obj.category):
                o = copy(obj)
                o.xy = (obj.x - self.offset_x + self.pads[0], obj.y - self.offset_y + self.pads[1])
                self.objects.append(o)

        return self.obs(new_obs.moveaxis(0, -1).numpy())

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        img = F.pad(torch.tensor(obs).moveaxis(-1, 0), self.padding, self.fill, self.padding_mode)
        self.offset_y, self.offset_x, self.h, self.w = super().get_params(img, self.size)

        # fill buffer
        self.observation(obs)
        self.observation(obs)
        self.observation(obs)

        return self.observation(obs), info

    def __getattr__(self, name: str):
        return super(ObservationWrapper, self).__getattr__(name)


class ShuffleObjectsWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        np.random.shuffle(observation)
        return observation


class LandmarkWrapper(gym.ObservationWrapper):

    class Landmark(GameObject):
        def __init__(self, x, y):
            super().__init__()
            self.xy = (x, y)


    def __init__(self, env, landmarks):
        super().__init__(env)
        self.objects = []
        self.landmarks = [LandmarkWrapper.Landmark(*landmark) for landmark in landmarks]
        self.max_objs = get_max_objects(env.game_name, env.hud)  # noqa: type(env) == OCAtari
        self.max_objs["Landmark"] = len(landmarks)

    def observation(self, observation):
        self.objects = self.env.objects + self.landmarks
        return observation


# object feature functions
def dx_dy_center(o, x, y):
    center_x, center_y = o.center
    return o.dx / x, o.dy / y, center_x / x, center_y / y


def w_h_dx_dy_center(o, x, y):
    center_x, center_y = o.center
    w, h = o.wh
    return w / y, h / x, o.dx / x, o.dy / y, center_x / x, center_y / y
