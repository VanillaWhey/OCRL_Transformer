# Supplementary Material

This is the supplementary material for the TMLR submission _Anchored Object Representation for Reinforcement Learning_.

This directory contains the source code required to replicate our experiments, along with a GIF of ORBiT playing Foosball.

## Repository Structure

```
.
├── atari/      # ORBiT and baselines on the Atari Learning Environment
├── foosball/   # ORBiT and baselines on the Foosball robotic task
└── gif/        # Foosball rollout video referenced below
```

Each experiment folder is self-contained, with its own installation instructions, dependencies, and usage examples.

## Atari Learning Environment

We evaluate ORBiT on the Atari Learning Environment, building on an object-centric observation wrapper and a PPO training loop. Source code and instructions for the Atari experiments are in the [`atari`](atari/README.md) folder.

## Foosball

We evaluate ORBiT on a simulated robotic Foosball task built on Isaac Sim, comparing ORBiT against an object-centric transformer baseline (OCT) and a standard PPO baseline. Source code, configuration files, and pretrained checkpoints for the Foosball experiments are in the [`foosball`](foosball/README.md) folder.

### Foosball: ORBiT 3 vs. ORBiT 2 (slowed down 3x)

![Foosball GIF](gif/foosball.gif)
