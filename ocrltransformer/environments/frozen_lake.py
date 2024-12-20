import numpy as np
import gymnasium as gym

from gymnasium.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv


class OCFrozenLake(gym.ObservationWrapper):
    metadata = FrozenLakeEnv.metadata

    def __init__(self, **kwargs):
        super().__init__(env=gym.make("FrozenLake-v1",
                                      is_slippery=False, **kwargs))
        self.observation_space = gym.spaces.Box(0, 100,(16, 5))
        self.base_state = None
        self.player_idx = 0

    def observation(self, observation):
        self.base_state[self.player_idx, 3] = observation % 4
        self.base_state[self.player_idx, 4] = observation // 4

        return self.base_state

    def reset(self, *, seed=None, options=None):
        if options and "desc" in options:
            self.env.unwrapped.desc = options["desc"]
        else:
            size = int(self.env.np_random.integers(3, 6, 1))
            self.env.unwrapped.desc = generate_random_map(size=size)

        self.base_state = []
        for row in range(len(self.env.unwrapped.desc)):
            for cul in range(len(self.env.unwrapped.desc[0])):
                if self.env.unwrapped.desc[row][cul] == "H":
                    self.base_state.append([0, 1, 0, row, cul])
                elif self.env.unwrapped.desc[row][cul] == "S":
                    self.player_idx = len(self.base_state)
                    self.base_state.append([1, 0, 0, row, cul])
                elif self.env.unwrapped.desc[row][cul] == "G":
                    self.base_state.append([0, 0, 1, row, cul])

        for i in range(len(self.base_state), 16):
            self.base_state.append([0]*5)
        self.base_state = np.asarray(self.base_state, dtype=np.float32)

        return super().reset(seed=seed, options=options)