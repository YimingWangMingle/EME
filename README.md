# Effective Metric-Based Exploration Bonus (EME)
This repository contains the code for the NeurIPS 2024 paper
[Rethinking Exploration in Reinforcement Learning with Effective Metric-Based Exploration Bonus]()

### Overview:
The EME model introduces an effective metric-based exploration bonus that addresses the inherent limitations and approximation inaccuracies of current metric-based state discrepancy methods for exploration. Specifically, it proposes a robust metric for state discrepancy evaluation and a diversity-enhanced scale factor dynamically adjusted by the variance of prediction from an ensemble of reward models.

## Experimental Results

### Results of EME

![Our Method Performance 1](./figs/41.gif)
![Our Method Performance 2](./figs/42.gif)
![Our Method Performance 3](./figs/43.gif)
![Our Method Performance 4](./figs/44.gif)

## Installing Dependencies

To set up the environment and install dependencies, follow these steps:

Clone the repository and install basic requirements
```bash
git clone --recurse-submodules [EME]
cd EME
pip install -e .
pip install -r requirements.txt
```

Create a new conda environment and install required packages
```bash
conda create -n eme python=3.9 cmake=3.14.0
conda activate eme
conda install habitat-sim withbullet -c conda-forge -c aihabitat
```

Clone and install `habitat-lab` `habitat-baselines`
```bash
cd src
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e .
pip install -e habitat-baselines
```

Install additional dependencies
```bash
conda install git git-lfs
```

Download the required datasets
```bash
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
python -m habitat_sim.utils.datasets_download --uids mp3d_example_scene --data-path data/
python -m habitat_sim.utils.datasets_download --username xxxxxxxxxxxx --password xxxxxxxxxxxx --uids hm3d_minival_v0.2
```
Verify the installation by running an example
```bash
python src/habitat-lab/examples/example.py
```

## Test Experiments

After successfully installing `habitat-lab` and `habitat-sim` dependencies, you can run a test experiment to ensure everything is set up correctly.

### Running the Training Script

Navigate to the `habitat-lab` directory and execute the `eme_train.py` script:

```bash
cd src/habitat-lab
python eme_train.py
```

## Other Experiments
First, you will need to install the Habitat simulator. To do this, follow the instructions from the official Habitat repo [here](https://github.com/facebookresearch/habitat-lab), and make sure you can run the DD-PPO baseline. You will also need to download the [HM3D dataset](https://github.com/facebookresearch/habitat-matterport3d-dataset)

To run EME locally on a single machine, do:
```
cd ./habitat-lab/scripts
./run_local_reward_free.sh
```

This is useful for debugging, but is too slow otherwise. To run for enough steps, you will need to run distributed over multiple GPUs.

To run EME, ICM, RND, NovelD, E3B with 32 GPUs on a Slurm cluster, do:
```
sbatch multi_node_reward_free_{eme,icm, rnd, noveld,e3b}.sh
```
The implementation of other tasks will be released soon (up-to-date)

## Cite as
> Wang, Y., Zhao, K., & Liu, F. Rethinking Exploration in Reinforcement Learning with Effective Metric-Based Exploration Bonus. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

### Bibtex:
```
@inproceedings{wangrethinking,
  title={Rethinking Exploration in Reinforcement Learning with Effective Metric-Based Exploration Bonus},
  author={Wang, Yiming and Zhao, Kaiyan and Liu, Furui and others},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
  year={2024}
}
```

## Acknowledgements
This repository uses Habitat API (https://github.com/facebookresearch/habitat-api) Habitat-lab (https://github.com/facebookresearch/habitat-lab) Habitat-smi (https://github.com/facebookresearch/habitat-sim) and parts of the code from the habitat. We thank the contributors for their work.
