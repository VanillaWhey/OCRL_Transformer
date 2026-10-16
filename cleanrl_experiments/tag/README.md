# ORBiT Tag

This directory contains the object-centric Tag environment used in our reference-frame experiments. Training uses the same scripts as the Atari experiments, in the [`cleanrl`](../cleanrl) folder.

## Environment

The agent (Player) moves in a walled box and has to catch a Goal as often as it can within 500 steps. Each catch gives a reward of +1, after which the Goal and all Distractors re-spawn away from the agent. The Distractors move exactly like the Goal, so the agent can tell them apart only by their category. By default the box contains 5 Distractors and 2 static Landmarks, and the Goal flees from the agent.

Each observation has one row per object: a one-hot category (Player, Goal, Distractor, Landmark) followed by its kinematic state. The action is a continuous `(throttle, turn)` pair in `[-1, 1]^2`.

| File | Contents |
| --- | --- |
| [`tag_box_env.py`](tag_box_env.py) | Walled-box Tag environment used in the paper (`TagBoxVecEnv`, `TagBoxEnv`) |
| [`tag_env.py`](tag_env.py) | Torus Tag environment that the box version is derived from; it also defines the observation layout |
| [`wrappers.py`](wrappers.py) | Reference-frame wrappers (e.g. `VecKinematicFrame`) and the adapter to the training loop |
| [`tag_setup.py`](tag_setup.py) | Experimental setup: worlds, reference-frame conditions, `make_vec_env` and `evaluate` |
| [`policies.py`](policies.py) | Reference policies (`random`, `chase`) and the evaluation loop |

### Reference frames

The anchor of the agent's reference frame is set with `--player-name`:

| `--player-name` | Observation |
| --- | --- |
| *(empty, default)* | Absolute world coordinates |
| `agent` | Co-moving, co-rotating frame of the agent |
| `goal` | Co-moving, co-rotating frame of the goal |
| `distractor` | Co-moving, co-rotating frame of the first distractor |
| `landmark` | Co-moving, co-rotating frame of the first landmark |

## Installation

Install the requirements of the [`cleanrl_experiments`](../README.md) folder:
```bash
pip install -r cleanrl_experiments/requirements.txt
```

## Usage

Run all commands from the repository root, since the code is imported as the `cleanrl_experiments` package.

### Training

The environment ID has the form `tag/<world>-v0`, where `<world>` is `box` (main results, 5 distractors) or `box_heldout` (3 distractors). To train ORBiT in the agent's reference frame, run:
```bash
python -m cleanrl_experiments.cleanrl.ppo_atari_orbit --seed 0 --env-id tag/box-v0 --exp-name ORBiT_Tag_agent_0 --player-name agent
```

To train the PPObj baseline (MLP over the flattened object list), run:
```bash
python -m cleanrl_experiments.cleanrl.ppo_atari_obj --seed 0 --env-id tag/box-v0 --exp-name PPObj_Tag_agent_0 --player-name agent
```

The number of distractors can be changed with `--num-distractors`. During training, an episode ends at the first catch or after 500 steps. At the end of training, the model is saved to `./wandb/runs/tag/<world>-v0__<exp-name>__<seed>__<timestamp>/<exp-name>.cleanrl_model`, or to the W&B run directory with `--track`. The model is then evaluated as described below.

### Evaluation

Evaluation runs the deterministic policy on 100 full 500-step episodes, in which the Goal re-spawns after every catch. It reports catches per episode for the agent and for the `random` and `chase` reference policies, along with the normalized score `(agent - random) / (chase - random)`. A score of 0 matches the random policy, and 1 matches `chase`, which steers straight at the Goal at full speed. Scores above 1 are possible.

To evaluate a trained model, run:
```bash
python -m cleanrl_experiments.eval.eval_tag <path/to/model.cleanrl_model> [n_distractors]
```

The optional `n_distractors` argument evaluates the model with a different number of distractors than in training. This works only for ORBiT; PPObj needs the number of objects it was trained with.

The scores of the reference policies are cached in `tag/baselines.json` the first time they are computed.
