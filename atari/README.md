# ORBiT Atari

This directory contains the learning script for our Atari experiments in the folder `cleanrl`. This script is based on the repository https://github.com/vwxyzjn/cleanrl.
Further, it contains our main code for the ORBiT architecture and OARC wrapper in the folder `orbit`.

Please install all the requirements with pip:
````bash
pip install -r requirements.txt
````

To start a training run on Asterix, run:
````bash
python cleanrl/ppo_atari_orbit.py --seed 0 --env-id ALE/Asterix-v5 --exp-name ORBiT_Asterix_0 --obs_mode dqn --player_name Player
````