**Status:** Archive (code is provided as-is, no updates expected)

# Multi-Agent Particle Environment

A simple multi-agent particle world with a continuous observation and discrete action space, along with some basic simulated physics.
Used in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).

## Getting started:

- To install, `cd` into the root directory and type `pip install -e .`

- To interactively view moving to landmark scenario (see others in ./scenarios/):
`bin/interactive.py --scenario simple.py`

- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), numpy (1.14.5)

- To use the environments, look at the code for importing them in `make_env.py`.


