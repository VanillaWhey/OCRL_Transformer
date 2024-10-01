from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.core import WrapperObsType, seeding

from ocatari import OCAtari

class OCMultiGame(gym.Wrapper):
    def __init__(self, *env_strs, exclude="", wrappers=(), **kwargs):
        assert len(env_strs) > 0

        self.exclude_env = OCAtari(env_name=exclude, gym_args=dict(full_action_space=True), **kwargs)
        self.envs = [OCAtari(env_name=env_str, gym_args=dict(full_action_space=True), **kwargs) for env_str in env_strs if env_str != exclude]

        # build reference list with max objects
        ref_dict = {}
        for env in self.envs + [self.exclude_env]:
            if env.hud:
                obj_dict = env.ram_module_name.MAX_NB_OBJECTS_HUD
            else:
                obj_dict = env.ram_module_name.MAX_NB_OBJECTS
            for obj in obj_dict.keys():
                if obj in ref_dict:
                    ref_dict[obj] = max(obj_dict[obj], ref_dict[obj])
                else:
                    ref_dict[obj] = obj_dict[obj]
        self.reference_list = []
        for o in ref_dict.keys():
            self.reference_list.extend([o for _ in range(ref_dict[o])])

        for env in self.envs:
            env.reference_list = self.reference_list

        # wrap all envs in the wrappers
        for i in range(len(self.envs)):
            for wrapper in wrappers:
                self.envs[i] = wrapper(self.envs[i])

        super().__init__(self.envs[0])

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        # randomly select new env
        if options.get('test', False):
            self.env = self.exclude_env
        else:
            self.env = self.envs[self.np_random.integers(len(self.envs))]

        return super().reset(seed=seed, options=options)

