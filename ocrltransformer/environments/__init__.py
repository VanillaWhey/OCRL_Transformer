import gymnasium as gym

from.acrobot import OCAcrobot
from .cart_pole import OCCartPole
from .frozen_lake import OCFrozenLake
from .taxi import OCTaxi


gym.envs.register(
     id='OCAcrobot-v0',
     entry_point='ocrltransformer.environments:OCAcrobot',
)

gym.envs.register(
     id='OCCartPole-v0',
     entry_point='ocrltransformer.environments:OCCartPole',
)

gym.envs.register(
     id='OCFrozenLake-v0',
     entry_point='ocrltransformer.environments:OCFrozenLake',
)

gym.envs.register(
     id='OCTaxi-v0',
     entry_point='ocrltransformer.environments:OCTaxi',
)
