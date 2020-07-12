Plan-to-Explore implementation in PyTorch
======

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

## Plan-to-explore
This repo implements the Plan-to-explore algorithm from [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960) based on the [PlaNet-Pytorch](https://github.com/Kaixhin/PlaNet). It has been confirmed working on the DeepMind Control Suite/MuJoCo environment. Hyperparameters have been taken from the paper.

## Installation
To install all dependencies with Anaconda run using the following commands. 

`conda env create -f conda_env.yml` 

`source activate p2e` 

## Training (e.g. DMC walker-walk zero-shot)

Zero-shot
```bash
python main.py --algo p2e --env walker-walk --action-repeat 2 --id name-of-experiement --zero-shot
```

Few-shot
```bash
python main.py --algo p2e --env walker-walk --action-repeat 2 --id name-of-experiement
```

For best performance with DeepMind Control Suite, try setting environment variable `MUJOCO_GL=egl` (see instructions and details [here](https://github.com/deepmind/dm_control#rendering)).

Use Tensorboard to monitor the training.

You can see the performance from the zero-shot/few-shot trained policy on the `test/episode_reward`.

`tensorboard --logdir results`



## Links
- [Planning to Explore via Self-Supervised World Models](https://ramanans1.github.io/plan2explore/)
- [ramanans/plan2explore](https://github.com/ramanans1/plan2explore)
