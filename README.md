# Effective Metric-Based Exploration Bonus (EME)
This repository contains the code for the NeurIPS 2024 submission:
[Rethinking Exploration in Reinforcement Learning with Effective Metric-Based Exploration Bonus]()

### Overview:
The EME model introduces an effective metric-based exploration bonus that addresses the inherent limitations and approximation inaccuracies of current metric-based state discrepancy methods for exploration. Specifically, it proposes a robust metric for state discrepancy evaluation and a diversity-enhanced scale factor dynamically adjusted by the variance of prediction from an ensemble of reward models.

## Experimental Results

### Results of EME
The following images showcase the performance of our proposed Effective Metric-Based Exploration Bonus (EME) algorithm in various scenarios:

![Our Method Performance 1](./figs/41.gif)
![Our Method Performance 2](./figs/42.gif)
![Our Method Performance 3](./figs/43.gif)
![Our Method Performance 4](./figs/44.gif)

### Results of Other Methods
For comparison, we also present the performance of other methods (see details in the paper) under the same experimental conditions. These comparisons highlight the advantages of EME in handling complex exploration tasks.

![Other Method Performance 1](./figs/11.gif)
![Other Method Performance 2](./figs/12.gif)
![Other Method Performance 3](./figs/13.gif)
![Other Method Performance 4](./figs/14.gif)

![Other Method Performance 5](./figs/21.gif)
![Other Method Performance 6](./figs/22.gif)
![Other Method Performance 7](./figs/23.gif)
![Other Method Performance 8](./figs/24.gif)

![Other Method Performance 9](./figs/31.gif)
![Other Method Performance 10](./figs/32.gif)
![Other Method Performance 11](./figs/33.gif)
![Other Method Performance 12](./figs/34.gif)

It is evident that our method significantly improves diversity and exploration efficiency. The EME model introduces a more robust metric for state discrepancy evaluation and dynamically adjusts the diversity-enhanced scale factor by the variance of prediction from an ensemble of reward models, effectively increasing the breadth and efficiency of state exploration. These results validate the effectiveness and advantages of our method in handling exploration tasks.


## Installing Dependencies

To set up the environment and install dependencies, follow these steps:

### Clone the Repository and Install Basic Requirements
```bash
git clone --recurse-submodules [EME]
cd EME
pip install -e .
pip install -r requirements.txt
```

### Create a New Conda Environment and Install Required Packages
```bash
conda create -n eme python=3.9 cmake=3.14.0
conda activate eme
conda install habitat-sim withbullet -c conda-forge -c aihabitat
```

### Clone and Install `habitat-lab` `habitat-baselines`
```bash
cd src
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e .
pip install -e habitat-baselines
```

### Install Additional Dependencies
```bash
conda install git git-lfs
```

### Download the Required Datasets
```bash
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
python -m habitat_sim.utils.datasets_download --uids mp3d_example_scene --data-path data/
python -m habitat_sim.utils.datasets_download --username xxxxxxxxxxxx --password xxxxxxxxxxxx --uids hm3d_minival_v0.2
```

### Verify the Installation by Running an Example
```bash
python src/habitat-lab/examples/example.py
```

## Run Experiments
```
cd src/algos/EME/
python -u train.py --env-name='SolariesNoFrameskip-v4' --cuda (if cuda is available) --lr-decay  --log-dir='logs' --seed=123
```

## Cite as
> Anonymous, Rethinking Exploration in Reinforcement Learning with Effective Metric-Based Exploration Bonus. NeurIPS submission 2024.

### Bibtex:
```
@inproceedings{anonymous2024rethinking,
  title={Rethinking Exploration in Reinforcement Learning with Effective Metric-Based Exploration Bonus},
  author={Anonymous},
  booktitle={Anonymous},
  year={2024}
}
```

## Acknowledgements
This repository uses Habitat API (https://github.com/facebookresearch/habitat-api) Habitat-lab (https://github.com/facebookresearch/habitat-lab) Habitat-smi (https://github.com/facebookresearch/habitat-sim) and parts of the code from the habitat. We thank the contributors for their work.
