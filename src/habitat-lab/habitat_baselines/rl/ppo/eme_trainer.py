# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import pdb
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from gym import spaces
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from habitat import Config, VectorEnv, logger
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetPolicy,
    PointNavResNetNet,
    InverseDynamicsNet,
)
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    action_to_velocity_control,
    batch_obs,
    generate_video,
)
from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.utils.tensorboard import TensorboardWriter

EPSILON = 1e-9


def _sqrt(x, tol=0.):
    # Computes the square root while ensuring non-negative input
    tol = torch.zeros_like(x)
    return torch.sqrt(torch.maximum(x, tol))


def cosine_distance(x, y):
    # Computes the cosine distance between two tensors
    numerator = torch.sum(x * y, dim=-1, keepdim=True)
    denominator = torch.sqrt(
        torch.sum(x.pow(2.), dim=-1, keepdim=True)) * torch.sqrt(torch.sum(y.pow(2.), dim=-1, keepdim=True))
    cos_similarity = numerator / (denominator + EPSILON)

    return torch.atan2(_sqrt(1. - cos_similarity.pow(2.)), cos_similarity)


def compute_kl_divergence(policy_i, policy_j):
    """
    Compute the KL divergence between two policies.
    """
    # Compute the log probabilities for policy_i and probabilities for policy_j
    policy_i_log_probs = F.log_softmax(policy_i, dim=-1)
    policy_j_probs = F.softmax(policy_j, dim=-1)
    kl_div = F.kl_div(policy_i_log_probs, policy_j_probs, reduction='batchmean')
    return kl_div


class StateRewardDecoder(nn.Module):
    def __init__(self, encoder_feature_dim, max_sigma=1e0, min_sigma=1e-4):
        super().__init__()
        # Define the neural network architecture for reward decoding
        self.trunck = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 2))

        # Define maximum and minimum bounds for sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

    def forward(self, x):
        # Forward pass through the network to get mu and sigma
        y = self.trunck(x)
        sigma = y[..., 1:2]
        mu = y[..., 0:1]
        sigma = torch.sigmoid(sigma)  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma
       
    def loss(self, mu, sigma, r, reduce='mean'):
        # Compute the loss between predicted rewards and actual rewards
        diff = (mu - r.detach()) / sigma
        if reduce == 'none':
            loss = 0.5 * (0.5 * diff.pow(2) + torch.log(sigma))
        elif  reduce =='mean':
            loss = 0.5 * torch.mean(0.5 * diff.pow(2) + torch.log(sigma))
        else:
            raise NotImplementedError

        return loss


class EMETrainer(BaseRLTrainer):
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []

        # Set the device to CUDA if available, else use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load hyperparameters from the config
        self.kl_coef = config.RL.EME.kl_coef if config else 0.01
        self.gamma = config.RL.PPO.gamma if config else 0.99
        self.max_reward_scaling = config.RL.EME.max_reward_scaling if config else 1.0

        # Initialize the state reward decoder model
        self.state_reward_decoder = StateRewardDecoder(encoder_feature_dim=config.RL.EME.encoder_feature_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.state_reward_decoder.parameters()),
            lr=config.RL.EME.decoder_lr if config else 1e-3
        )

        # Initialize ensemble of reward models for diversity-enhanced scaling factor
        self.num_ensemble_models = config.RL.EME.num_ensemble_models if config else 5
        self.reward_models = nn.ModuleList([
            StateRewardDecoder(encoder_feature_dim=config.RL.EME.encoder_feature_dim).to(self.device)
            for _ in range(self.num_ensemble_models)
        ])
        self.ensemble_optimizer = torch.optim.Adam(
            [param for model in self.reward_models for param in model.parameters()],
            lr=config.RL.EME.decoder_lr if config else 1e-3
        )

        self._init_train()

    def _init_train(self):
        # Initialize the training environment, actor-critic, and rollouts
        self._init_envs()
        self._setup_actor_critic_agent()
        self.rollouts = RolloutStorage(
            self.config.RL.PPO.num_steps,
            self.envs.num_envs,
            self.obs_space,
            self.policy_action_space,
            self.config.RL.PPO.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=self.config.RL.PPO.use_double_buffered_sampler,
        )
        self.rollouts.to(self.device)
        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )

    def _setup_actor_critic_agent(self):
        # Set up the actor-critic and PPO agent
        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config(
            self.config, observation_space, self.policy_action_space
        )
        self.actor_critic.to(self.device)

        # Set up PPO agent with appropriate parameters
        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=self.config.RL.PPO.clip_param,
            ppo_epoch=self.config.RL.PPO.ppo_epoch,
            num_mini_batch=self.config.RL.PPO.num_mini_batch,
            value_loss_coef=self.config.RL.PPO.value_loss_coef,
            entropy_coef=self.config.RL.PPO.entropy_coef,
            lr=self.config.RL.PPO.lr,
            eps=self.config.RL.PPO.eps,
            max_grad_norm=self.config.RL.PPO.max_grad_norm,
            use_normalized_advantage=self.config.RL.PPO.use_normalized_advantage,
        )

    def _collect_rollout_step(self):
        # Collect rollout step by computing actions and stepping the environment
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    def _compute_actions_and_step_envs(self):
        # Compute actions for all environments and step them
        num_envs = self.envs.num_envs

        with torch.no_grad():
            # Get the current step batch from rollouts
            step_batch = self.rollouts.buffers[self.rollouts.current_rollout_step_idx]

            # Use actor-critic to compute values, actions, and log probabilities
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

            # Compute intrinsic reward using EME metric
            eme_rewards = self._compute_eme_metric(step_batch)

        # Add intrinsic reward to external reward and update rollouts
        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            rewards=eme_rewards,
        )

    def _compute_eme_metric(self, step_batch):
        """
        Compute the EME metric for intrinsic reward calculation.
        """
        # Extract observations, actions, and rewards from the step batch
        observations = step_batch["observations"]
        next_observations = step_batch["next_observations"]
        actions = step_batch["actions"]
        rewards = step_batch["rewards"]

        # Get embeddings from the actor-critic model
        embed_i = self.actor_critic.net(observations)
        embed_j = self.actor_critic.net(next_observations)

        # Compute cosine similarity distance
        cosine_dist = cosine_distance(embed_i, embed_j)

        # Compute KL divergence between current and next policy
        policy_i = self.actor_critic(observations)
        policy_j = self.actor_critic(next_observations)
        kl_div = compute_kl_divergence(policy_i, policy_j)

        # Compute reward prediction loss using StateRewardDecoder
        reward_mu, reward_sigma = self.state_reward_decoder(embed_i)
        reward_loss = self.state_reward_decoder.loss(reward_mu, reward_sigma, rewards)

        # Compute diversity-enhanced scaling factor using ensemble variance
        ensemble_predictions = torch.stack([model(embed_i)[0] for model in self.reward_models], dim=0)
        ensemble_mean = ensemble_predictions.mean(dim=0)
        ensemble_variance = torch.mean((ensemble_predictions - ensemble_mean) ** 2, dim=0)
        diversity_scaling_factor = torch.clamp(ensemble_variance, max=self.max_reward_scaling)

        # Combine metrics to get EME reward
        eme_metric = cosine_dist + self.kl_coef * kl_div + reward_loss + diversity_scaling_factor
        return eme_metric

    def train(self):
        """
        Main training loop for EME.
        """
        for _ in range(self.config.RL.PPO.num_updates):
            self._collect_rollout_step()
            self._update_agent()

    def _update_agent(self):
        # Update the PPO agent using rollouts
        with torch.no_grad():
            step_batch = self.rollouts.buffers[self.rollouts.current_rollout_step_idx]
            next_value = self.actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )
        self.rollouts.compute_returns(next_value, self.gamma, self.config.RL.PPO.tau)
        self.agent.train()
        value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)
        self.rollouts.after_update()

if __name__ == "__main__":
    config = Config() 
    eme_trainer = EMETrainer(config)
    eme_trainer.train()
