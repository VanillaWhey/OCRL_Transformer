# ORBiT Atari

This directory contains the code for our Atari and Tag experiments. The training scripts are in the [`cleanrl`](cleanrl) folder and are based on the [CleanRL](https://github.com/vwxyzjn/cleanrl) repository: [`ppo_atari_orbit.py`](cleanrl/ppo_atari_orbit.py) contains the ORBiT architecture, and [`ppo_atari_obj.py`](cleanrl/ppo_atari_obj.py) contains the PPObj baseline. The OARC observation wrapper is in the [`atari`](atari) folder. The Tag environment is in the [`tag`](tag) folder and has its own [README](tag/README.md).

## Installation

Install all requirements with pip:
```bash
pip install -r cleanrl_experiments/requirements.txt
```

## Usage

Run all commands from the repository root, since the code is imported as the `cleanrl_experiments` package. To start an ORBiT training run on Asterix, run:
```bash
python -m cleanrl_experiments.cleanrl.ppo_atari_orbit --seed 0 --env-id ALE/Asterix-v5 --exp-name ORBiT_Asterix_0 --obs-mode dqn --player-name Player
```

To list all options, including the ORBiT hyperparameters and W&B tracking (`--track`), run:
```bash
python -m cleanrl_experiments.cleanrl.ppo_atari_orbit --help
```
