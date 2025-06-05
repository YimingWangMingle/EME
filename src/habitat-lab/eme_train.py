
import habitat_sim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import cv2
import os
import imageio
from datetime import datetime

# EME hyperparameters
learning_rate = 3e-4
gamma = 0.99
epsilon_clip = 0.2
K_epochs = 4
update_timestep = 2000
max_timesteps = 1000000
max_steps_per_episode = 500
kl_coef = 0.01
max_reward_scaling = 1.0
encoder_feature_dim = 512
decoder_lr = 1e-3
num_ensemble_models = 5
M = 1.0  # Maximum reward scaling factor

data_path = "/home/xxx/habitat-lab/"
test_scene = os.path.join(data_path, "data/xxx/xxx/xxx.glb") # Change to your own path

sim_settings = {
    "width": 256,
    "height": 256,
    "scene": test_scene,
    "default_agent": 0,
    "sensor_height": 1.5,
    "color_sensor": True,
    "depth_sensor": False,
    "semantic_sensor": False,
    "seed": 1,
    "enable_physics": False,
}

# Create Habitat environment
def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    sensor_specs = []
    if settings["color_sensor"]:
        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [settings["height"], settings["width"]]
        color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

# Visual feature extractor (CNN)
class VisualEncoder(nn.Module):
    def __init__(self, output_dim=encoder_feature_dim):
        super(VisualEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # Output: (32, 63, 63)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: (64, 30, 30)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Output: (64, 28, 28)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, output_dim),  # Output: (output_dim)
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)

# PPO network definition, incorporating visual feature extractor
class PPO(nn.Module):
    def __init__(self, action_space):
        super(PPO, self).__init__()
        self.visual_encoder = VisualEncoder()
        self.fc1 = nn.Linear(encoder_feature_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_layer = nn.Linear(128, action_space)
        self.value_layer = nn.Linear(128, 1)

    def forward(self, state):
        embed = self.visual_encoder(state)
        x = torch.relu(self.fc1(embed))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.action_layer(x), dim=-1)
        state_value = self.value_layer(x)
        return action_probs, state_value, embed

    def act(self, state):
        action_probs, state_value, embed = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), embed

    def evaluate(self, state, action):
        action_probs, state_value, embed = self.forward(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_value, dist_entropy, embed

# StateRewardDecoder definition
class StateRewardDecoder(nn.Module):
    def __init__(self, encoder_feature_dim, max_sigma=1e0, min_sigma=1e-4):
        super(StateRewardDecoder, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

    def forward(self, x):
        y = self.trunk(x)
        mu = y[..., 0:1]
        sigma = y[..., 1:2]
        sigma = torch.sigmoid(sigma)  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def loss(self, mu, sigma, r, reduce='mean'):
        diff = (mu - r.detach()) / sigma
        if reduce == 'none':
            loss = 0.5 * (diff.pow(2) + torch.log(sigma))
        elif reduce == 'mean':
            loss = 0.5 * torch.mean(0.5 * diff.pow(2) + torch.log(sigma))
        else:
            raise NotImplementedError
        return loss

# Memory pool definition
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.embeddings = []  # Save state embeddings

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.embeddings[:]

# Compute KL divergence
def compute_kl_divergence(action_probs_i, action_probs_j):
    """
    Compute the KL divergence between two policies.
    action_probs_i and action_probs_j are probabilities from the policy network.
    """
    dist_i = Categorical(probs=action_probs_i)
    dist_j = Categorical(probs=action_probs_j)
    kl_div = torch.distributions.kl_divergence(dist_i, dist_j)
    return kl_div

# Compute cosine distance
def cosine_distance(x, y):
    numerator = torch.sum(x * y, dim=-1, keepdim=True)
    denominator = torch.sqrt(torch.sum(x.pow(2.), dim=-1, keepdim=True)) * torch.sqrt(torch.sum(y.pow(2.), dim=-1, keepdim=True))
    cos_similarity = numerator / (denominator + 1e-9)
    return torch.atan2(torch.sqrt(1. - cos_similarity.pow(2)), cos_similarity)

def ppo_update(memory, policy, state_reward_decoder, reward_models, optimizer_policy, optimizer_decoder, optimizer_ensemble, device):
    # Convert data in memory to tensors
    rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device)
    states = torch.stack(memory.states).to(device)  # Shape: (N, 3, H, W)
    actions = torch.tensor(memory.actions, dtype=torch.int64).to(device)
    old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32).to(device)
    embeddings = torch.stack(memory.embeddings).to(device)  # Shape: (N, feature_dim)

    # Compute EME intrinsic reward (if enough transitions exist)
    if len(memory.states) >= 2:
        s_i = states[:-1]
        s_j = states[1:]
        embed_i = embeddings[:-1]
        embed_j = embeddings[1:]
        rewards_i = rewards[:-1]
        rewards_j = rewards[1:]

        #  cosine distance
        cosine_dist = cosine_distance(embed_i, embed_j)

        #  KL divergence
        with torch.no_grad():
            action_probs_i, _, _ = policy.forward(s_i)
            action_probs_j, _, _ = policy.forward(s_j)
        kl_div = compute_kl_divergence(action_probs_i, action_probs_j)

        #  reward prediction loss
        mu_i, sigma_i = state_reward_decoder(embed_i)
        mu_j, sigma_j = state_reward_decoder(embed_j)
        reward_loss = state_reward_decoder.loss(mu_i, sigma_i, rewards_i) + \
                      state_reward_decoder.loss(mu_j, sigma_j, rewards_j)

        #  ensemble variance
        with torch.no_grad():
            ensemble_predictions = torch.stack([model(embed_i)[0] for model in reward_models], dim=0)  # (K, N-1, 1)
            ensemble_mean = ensemble_predictions.mean(dim=0)
            ensemble_variance = torch.mean((ensemble_predictions - ensemble_mean) ** 2, dim=0)

        #  diversity scaling factor
        diversity_scaling_factor = torch.clamp(ensemble_variance, max=max_reward_scaling)

        #  EME intrinsic reward
        eme_metric = cosine_dist + kl_div + reward_loss + diversity_scaling_factor
        intrinsic_reward = eme_metric.squeeze(1) * torch.clamp(diversity_scaling_factor.squeeze(1), min=1.0, max=M)

        #  combine rewards
        combined_rewards = rewards[:-1] + intrinsic_reward.detach()

        #  PPO update with combined rewards
        states_short = states[:-1]
        actions_short = actions[:-1]
        old_logprobs_short = old_logprobs[:-1]

        for _ in range(K_epochs):
            logprobs, state_values, dist_entropy, _ = policy.evaluate(states_short, actions_short)
            state_values = torch.squeeze(state_values)

            advantages = combined_rewards - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            ratios = torch.exp(logprobs - old_logprobs_short)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(state_values, combined_rewards)
            entropy_loss = -dist_entropy.mean()

            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            optimizer_policy.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer_policy.step()

        #  optimize reward decoder
        decoder_loss = reward_loss.mean()
        optimizer_decoder.zero_grad()
        decoder_loss.backward()
        optimizer_decoder.step()

        #  optimize ensemble models
        ensemble_loss = 0
        for model in reward_models:
            pred_mu, _ = model(embed_i)
            ensemble_loss += nn.functional.mse_loss(pred_mu, rewards_i)
        ensemble_loss /= len(reward_models)

        optimizer_ensemble.zero_grad()
        ensemble_loss.backward()
        optimizer_ensemble.step()

    # Clear memory after update
    memory.clear_memory()


# Convert RGB image to BGR
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

# Generate random goal point
def generate_random_goal(sim):
    pathfinder = sim.pathfinder
    random_goal = pathfinder.get_random_navigable_point()
    return random_goal

# Calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Log messages to file
def log_to_file(log_file, log_message):
    with open(log_file, "a") as file:
        file.write(log_message + "\n")

# Main training loop
def train():
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = make_cfg(sim_settings)
    try:
        sim.close()
    except NameError:
        pass
    sim = habitat_sim.Simulator(cfg)

    # Initialize agent
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([0.0, 1.0, 0.0])
    agent.set_state(agent_state)

    # Set log file
    log_file = "training_log_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    log_to_file(log_file, "Episode | Steps | Total Reward | Distance to Goal | Reached Goal")

    # Generate random goal point
    goal_position = generate_random_goal(sim)
    print(f"Generated goal position: {goal_position}")

    # Define observation space and action space size
    obs_space = sim_settings["width"] * sim_settings["height"] * 3 
    action_space = 3 

    # Initialize PPO policy network
    policy = PPO(action_space).to(device)
    optimizer_policy = optim.Adam(policy.parameters(), lr=learning_rate)

    # Initialize StateRewardDecoder and ensemble of reward models
    state_reward_decoder = StateRewardDecoder(encoder_feature_dim=encoder_feature_dim).to(device)
    optimizer_decoder = optim.Adam(state_reward_decoder.parameters(), lr=decoder_lr)

    reward_models = nn.ModuleList([
        StateRewardDecoder(encoder_feature_dim=encoder_feature_dim).to(device)
        for _ in range(num_ensemble_models)
    ])
    optimizer_ensemble = optim.Adam(
        [param for model in reward_models for param in model.parameters()],
        lr=decoder_lr
    )

    memory = Memory()

    timestep = 0
    for t in range(max_timesteps):
        state = sim.get_sensor_observations()["color_sensor"]
        state = transform_rgb_bgr(state)
        state = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0  # Shape: (1, 3, H, W)
        done = False
        total_reward = 0
        steps = 0

        # Video writer
        video_writer = imageio.get_writer(f'episode_{t}_video.mp4', fps=20)

        while not done and steps < max_steps_per_episode:
            timestep += 1
            steps += 1

            # Select action
            action, log_prob, embed = policy.act(state)

            # Execute action
            if action == 0:
                sim.step("move_forward")
            elif action == 1:
                sim.step("turn_left")
            else:
                sim.step("turn_right")

            next_state = sim.get_sensor_observations()["color_sensor"]
            next_state = transform_rgb_bgr(next_state)
            next_state_tensor = torch.from_numpy(next_state).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            # Get current agent position
            agent_state = agent.get_state()
            current_position = agent_state.position

            # Calculate distance to goal
            distance_to_goal = calculate_distance(current_position, goal_position)

            # Assume closer distance gives higher reward, and end when reaching the goal
            extrinsic_reward = -distance_to_goal
            if distance_to_goal < 0.5:
                extrinsic_reward += 100 
                done = True

            # Check if maximum steps exceeded, force end if so
            if steps >= max_steps_per_episode:
                extrinsic_reward -= 50
                done = True

            # Save to memory
            memory.states.append(state.squeeze(0).cpu())
            memory.actions.append(action)
            memory.logprobs.append(log_prob.cpu().detach().numpy())
            memory.rewards.append(extrinsic_reward)
            memory.dones.append(done)
            memory.embeddings.append(embed.squeeze(0).cpu().detach())

            state = next_state_tensor
            total_reward += extrinsic_reward

            # Display real-time image
            cv2.imshow("Agent View", transform_rgb_bgr(sim.get_sensor_observations()["color_sensor"]))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Training interrupted by user")
                sim.close()
                return

            # Save current frame to video
            video_writer.append_data(transform_rgb_bgr(sim.get_sensor_observations()["color_sensor"]))

            # Update PPO and EME after reaching update interval
            if timestep % update_timestep == 0:
                ppo_update(memory, policy, state_reward_decoder, reward_models, optimizer_policy, optimizer_decoder, optimizer_ensemble, device)

        reached_goal = distance_to_goal < 0.5
        log_message = f"{t} | {steps} | {total_reward:.2f} | {distance_to_goal:.2f} | {reached_goal}"
        print(log_message)
        log_to_file(log_file, log_message)

        # Close video writer
        video_writer.close()

    sim.close()

# Start training
if __name__ == "__main__":
    train()
