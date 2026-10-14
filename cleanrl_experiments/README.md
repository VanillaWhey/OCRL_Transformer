# ORBiT Atari

This directory contains the code for our Atari experiments. The training script is in the [`cleanrl`](cleanrl) folder and is based on the [CleanRL](https://github.com/vwxyzjn/cleanrl) repository. The main code for the ORBiT architecture and the OARC wrapper is in the [`orbit`](atari) folder.

## Installation

Install all requirements with pip:
```bash
pip install -r requirements.txt
```

## Usage

Run all commands from the repository root (the code is imported as the `cleanrl_experiments` package). To start a training run on Asterix, run:
```bash
python -m cleanrl_experiments.cleanrl.ppo_atari_orbit --seed 0 --env-id ALE/Asterix-v5 --exp-name ORBiT_Asterix_0 --obs_mode dqn --player_name Player
```