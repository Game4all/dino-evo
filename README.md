
# `dino_evo`

This project aims to evolve a simple neural network agent using a genetic algorithm to play an infinite runner game such as the Google Chrome Dino.


## Installation & Usage

1. Start by creating a virtual environment and installing the requirements by `pip install -r requirements.txt`
2. Run a training pass using one of the following commands
```bash
# move to src directory
cd src/

# run a training run with training visualization enabled, with a population of 500 and 10 generations, with an exploration rate (alpha) of 0.6
python -m scripts.train --train-gui -p 500 -g 10 --alpha 0.6

# run the script with --help flag to get all usable flags
python -m scripts.train --help

```