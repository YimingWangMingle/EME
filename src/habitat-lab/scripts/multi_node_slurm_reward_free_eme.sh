#!/bin/bash
#SBATCH --job-name=ddppo-eme
#SBATCH --output=slurm/logs/%j.log
#SBATCH --error=slurm/logs/%j.err
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=4
#SBATCH --mem-per-cpu=5GB
#SBATCH --constraint=volta32gb
#SBATCH --partition=devlab
#SBATCH --time=72:00:00
#SBATCH --requeue

export MAGNUM_LOG=quiet

export MAGNUM_GPU_VALIDATION=ON
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}

module purge
module load cuda/11.0
module load cudnn/v8.0.3.33-cuda.11.0
module load NCCL/2.7.8-1-cuda.11.0

MAIN_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MAIN_ADDR

echo $name

set -x

entropy=0.000005
bonus_coef=0.01
kl_coef=0.1
max_reward_scaling=1.0
num_ensemble_models=5
seed=1
tag=hm3d-noreward-eme-rgbd-bc_${bonus_coef}-entropy_${entropy}-kl_${kl_coef}-mrs_${max_reward_scaling}-nem_${num_ensemble_models}-seed_${seed}
echo $tag

srun python -u -m habitat_baselines.run \
     --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_hm3d.yaml \
     --run-type train TENSORBOARD_DIR data/hm3d/tb/${tag} CHECKPOINT_FOLDER data/hm3d/ckpt/${tag} TASK_CONFIG.SEED ${seed} \
     TRAINER_NAME ddppo-eme \
     RL.PPO.entropy_coef $entropy \
     RL.EME.bonus_coef $bonus_coef \
     RL.EME.kl_coef $kl_coef \
     RL.EME.max_reward_scaling $max_reward_scaling \
     RL.EME.num_ensemble_models $num_ensemble_models \
     RL.EME.encoder_feature_dim 512 \
     TOTAL_NUM_STEPS 5e8 \
     NUM_UPDATES -1 \
     NUM_CHECKPOINTS 100
