# ORBiT Foosball
This repository is based on https://github.com/Jaykixx/Foosball with the permission to publish this modified code.
It contains all code to replicate the results presented in the paper, and we provide one trained model for each algorithm for testing.

## Installation
Please follow the installation instructions for [OmniIsaacGymEnvs](https://github.com/isaac-sim/OmniIsaacGymEnvs/tree/release/4.0.0) to set up the base environment.

## Usage
To run the various training scenarios locate the python executable in Isaac Sim as described in the installation tutorial for [OmniIsaacGymEnvs](https://github.com/isaac-sim/OmniIsaacGymEnvs/tree/release/4.0.0).

Navigate to the project folder.

### Training
To start training, run:

#### ORBiT
```bash
PYTHON_PATH main.py headless=true test=false checkpoint='' num_envs=4096 max_iterations=100_000 seed=3 task.env.stagnationPenalty=150 task.env.lossPenalty=150 task.env.winReward=150 task.env.terminationPenalty=150 task=FoosballMixedSelfPlay task.env.object_centric_obs=true train.params.network.mlp.units=[]
```

#### OCT
```bash
PYTHON_PATH main.py headless=true test=false checkpoint='' num_envs=4096 max_iterations=100_000 seed=3 task.env.stagnationPenalty=150 task.env.lossPenalty=150 task.env.winReward=150 task.env.terminationPenalty=150 task=FoosballMixedSelfPlay task.env.object_centric_obs=false train.params.network.mlp.units=[]
```

#### PPO
```bash
PYTHON_PATH main.py headless=true test=false checkpoint= num_envs=4096 max_iterations=100_000 seed=1 task.env.stagnationPenalty=150 task.env.lossPenalty=150 task.env.winReward=150 task.env.terminationPenalty=150 task=FoosballMixedSelfPlay task.env.object_centric_obs=false train.params.network.name=actor_critic train.params.network.mlp.units=[256,256,128]
```

The execution is otherwise identical to OmniIsaacGymEnvs. For more Information see [Link](https://github.com/isaac-sim/OmniIsaacGymEnvs/tree/release/4.0.0?tab=readme-ov-file#running-the-examples).

### Testing
To execute a test (here PPO vs. ORBiT), run:
```bash
PYTHON_PATH main.py headless=true seed=110 test=true num_envs=512 task=FoosballSelfPlay task.env.stagnationPenalty=0 task.env.terminationPenalty=0 task.env.lossPenalty=1 task.env.winReward=1 task.env.distToBallReward=false +task.env.goal_factor=2503 +train.params.config.player.games_num=5000 +train.params.config.player.max_goals=1000 train.params.network.name=oc_transformer train.params.network.name=actor_critic train.params.network.mlp.units=[256,256,128] task.env.object_centric_obs=false task.env.ball_relative_obs=true checkpoint="./models/ppo_1.pth" train.params.config.self_play_config.opponent.network.name=oc_transformer train.params.config.self_play_config.opponent.network.mlp.units=[] train.params.config.self_play_config.opponent.object_centric_obs=true train.params.config.self_play_config.opponent.ball_relative_obs=true train.params.config.self_play_config.opponent.checkpoint="./models/orbit_3.pth"
```