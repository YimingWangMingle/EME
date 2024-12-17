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
import torch.nn.functional as F
from geomloss import SamplesLoss  # Using GeomLoss to compute Wasserstein distance

# PPO hyperparameters
learning_rate = 3e-4
gamma = 0.99
epsilon_clip = 0.2
K_epochs = 4
update_timestep = 2000
max_timesteps = 1000000
max_steps_per_episode = 500  # Maximum number of steps per episode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulator settings
data_path = "/home/zky/Desktop/habitat-lab/"
test_scene = os.path.join(data_path, "data/hm3d/hm3d-val-habitat-v0.2/00800-TEEsavR23oF/TEEsavR23oF.basis.glb")

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

# Create Habitat environment configuration
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

# PPO network definition
class PPO(nn.Module):
    def __init__(self, action_space):
        super(PPO, self).__init__()
        # Assume the input image is 256x256x3, adjust convolution layers to match output size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Add adaptive average pooling
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Adjust based on convolution output
        self.action_layer = nn.Linear(512, action_space)
        self.value_layer = nn.Linear(512, 1)

    def forward(self, state):
        x = state.view(-1, 3, 256, 256)  # Adjust based on actual image size
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.adaptive_pool(x)  # Add adaptive average pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        action_probs = torch.softmax(self.action_layer(x), dim=-1)
        state_value = self.value_layer(x)
        return action_probs, state_value, x  # Return feature vector x

    def act(self, state):
        action_probs, state_value, features = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), features

    def evaluate(self, state, action):
        # This method is not used, as we use evaluate_from_features instead
        pass

    def evaluate_from_features(self, features, action):
        action_probs = torch.softmax(self.action_layer(features), dim=-1)
        state_value = self.value_layer(features)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_value, dist_entropy

# Bisimulation metric network definition
class BisimulationMetric(nn.Module):
    def __init__(self, state_dim):
        super(BisimulationMetric, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Output a single distance value

    def forward(self, state1, state2):
        x1 = torch.relu(self.fc1(state1))
        x1 = torch.relu(self.fc2(x1))
        x2 = torch.relu(self.fc1(state2))
        x2 = torch.relu(self.fc2(x2))
        distance = torch.abs(self.fc3(x1) - self.fc3(x2))
        return distance

# Inverse dynamic model definition
class InverseDynamicModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(InverseDynamicModel, self).__init__()
        self.fc1 = nn.Linear(state_dim * 2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_layer = nn.Linear(128, action_dim)

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_logits = self.action_layer(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        return action_probs

# Combined PPO and LIBERTY network definition
class PPOWithLiberty(nn.Module):
    def __init__(self, action_space, state_dim):
        super(PPOWithLiberty, self).__init__()
        self.policy = PPO(action_space)
        self.bisim_metric = BisimulationMetric(state_dim)
        self.inverse_dyn = InverseDynamicModel(state_dim, action_space)

    def forward(self, state):
        return self.policy(state)

# Experience replay memory definition
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]

# Convert RGB image to BGR
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

# Generate a random goal point
def generate_random_goal(sim):
    pathfinder = sim.pathfinder
    random_goal = pathfinder.get_random_navigable_point()
    return random_goal

# Calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Write log to file
def log_to_file(log_file, log_message):
    with open(log_file, "a") as file:
        file.write(log_message + "\n")

# PPO update function
def ppo_update(memory, policy, optimizer, inverse_optimizer, bisim_optimizer, action_space):
    # Compute discounted rewards
    rewards = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.dones)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        rewards.insert(0, discounted_reward)

    rewards = torch.tensor(rewards, dtype=torch.float32).detach().to(device)
    
    # Convert memory.states to a single NumPy array and create tensor
    states = torch.tensor(np.array(memory.states), dtype=torch.float32).to(device)  # shape [batch_size, 512]
    actions = torch.tensor(memory.actions, dtype=torch.int64).to(device)
    old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(device)

    # Normalize rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    for _ in range(K_epochs):
        # PPO loss
        logprobs, state_values, dist_entropy = policy.policy.evaluate_from_features(states, actions)
        state_values = torch.squeeze(state_values)

        advantages = rewards - state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(logprobs - old_logprobs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
        ppo_loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values, rewards) - 0.01 * dist_entropy

        # Calculate bisim_loss and inverse_loss
        if len(memory.states) >= 2:
            state_batch = states[:-1]  # [batch_size-1, 512]
            next_state_batch = states[1:]  # [batch_size-1, 512]
            r_i = rewards[:-1]
            r_j = rewards[1:]
            reward_diff = torch.abs(r_i - r_j).unsqueeze(1)  # [batch_size-1, 1]

            # Compute W2 distance using GeomLoss
            loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05)
            W2_distance = loss_fn(state_batch, next_state_batch)

            # Compute inverse dynamic model predictions
            inverse_action_probs = policy.inverse_dyn(state_batch, next_state_batch)  # [batch_size-1, action_space]
            actions_batch = actions[:-1]  # [batch_size-1]

            # Compute inverse bisimulation distance ||I(s_i, s_{i+1}) - I(s_j, s_{j+1})||_1
            actual_action_one_hot = F.one_hot(actions_batch, num_classes=action_space).float()  # [batch_size-1, action_space]
            inverse_bisim_distance = torch.norm(inverse_action_probs - actual_action_one_hot, p=1, dim=1).unsqueeze(1)  # [batch_size-1, 1]

            # Compute d_inv target value
            d_inv_target = reward_diff + (gamma * W2_distance) + (gamma * inverse_bisim_distance)

            # Compute d_inv predicted value
            d_inv_pred = policy.bisim_metric(state_batch, next_state_batch)

            # Compute bisim_loss
            bisim_loss = F.mse_loss(d_inv_pred, d_inv_target.detach())

            # Compute inverse_loss
            inverse_loss = F.cross_entropy(inverse_action_probs, actions_batch)

            # Total loss
            total_loss = ppo_loss.mean() + 0.1 * bisim_loss + 0.1 * inverse_loss
        else:
            # Only PPO loss
            total_loss = ppo_loss.mean()

        # Backpropagation and optimization
        optimizer.zero_grad()
        if len(memory.states) >= 2:
            bisim_optimizer.zero_grad()
            inverse_optimizer.zero_grad()
        total_loss.backward()
        if len(memory.states) >= 2:
            bisim_optimizer.step()
            inverse_optimizer.step()
        nn.utils.clip_grad_norm_(policy.policy.parameters(), 0.5)
        optimizer.step()

    # Clear memory
    memory.clear_memory()

# Main training loop
def train():
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

    # Generate random goal position
    goal_position = generate_random_goal(sim)
    print(f"Generated goal position: {goal_position}")

    # Define action space and state dimension
    action_space = 3  # Three actions: move forward, turn left, turn right
    state_dim = 512  # Feature dimension from PPO network's fc1 layer

    # Initialize PPOWithLiberty network and optimizers
    policy = PPOWithLiberty(action_space, state_dim).to(device)
    optimizer = optim.Adam(policy.policy.parameters(), lr=learning_rate)
    bisim_optimizer = optim.Adam(policy.bisim_metric.parameters(), lr=learning_rate)
    inverse_optimizer = optim.Adam(policy.inverse_dyn.parameters(), lr=learning_rate)
    memory = Memory()

    timestep = 0
    for t in range(max_timesteps):
        state = sim.get_sensor_observations()["color_sensor"]
        state = transform_rgb_bgr(state).flatten()

        # Get initial feature vector
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        _, _, initial_features = policy.policy(state_tensor)
        initial_features = initial_features.detach()

        done = False
        total_reward = 0
        steps = 0

        # Video writer
        video_writer = imageio.get_writer(f'episode_{t}_video.mp4', fps=20)

        while not done:
            timestep += 1
            steps += 1
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

            # Choose action
            action, log_prob, features = policy.policy.act(state_tensor)

            # Execute action
            if action == 0:
                sim.step("move_forward")
            elif action == 1:
                sim.step("turn_left")
            else:
                sim.step("turn_right")

            next_state = sim.get_sensor_observations()["color_sensor"]
            next_state = transform_rgb_bgr(next_state).flatten()

            # Get agent's current position
            agent_state = agent.get_state()
            current_position = agent_state.position

            # Calculate distance to goal
            distance_to_goal = calculate_distance(current_position, goal_position)

            # Basic reward
            reward = -distance_to_goal

            # Check if goal is reached
            if distance_to_goal < 0.5:  # End task if distance is less than a threshold
                reward += 100  # Give a large reward
                done = True

            # Check if maximum steps are exceeded, end task
            if steps >= max_steps_per_episode:
                done = True
                reward -= 50  # Give a penalty for not completing within time limit

            # Save to memory
            # features is [batch_size, state_dim]
            memory.states.append(features.detach().cpu().squeeze(0).numpy())  # Add .detach() and squeeze
            memory.actions.append(action)
            memory.logprobs.append(log_prob.item())
            memory.rewards.append(reward)
            memory.dones.append(done)

            state = next_state
            total_reward += reward

            # Display real-time image (optional, recommended only for debugging)
            cv2.imshow("Agent View", transform_rgb_bgr(sim.get_sensor_observations()["color_sensor"]))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Training interrupted by user")
                return

            # Save current frame to video
            frame = transform_rgb_bgr(sim.get_sensor_observations()["color_sensor"])
            video_writer.append_data(frame)

            # Update PPO after reaching the timestep interval
            if timestep % update_timestep == 0:
                ppo_update(memory, policy, optimizer, inverse_optimizer, bisim_optimizer, action_space)
                memory.clear_memory()

        reached_goal = distance_to_goal < 0.5
        log_message = f"{t} | {steps} | {total_reward:.2f} | {distance_to_goal:.2f} | {reached_goal}"
        print(log_message)
        log_to_file(log_file, log_message)

        # Close video writer
        video_writer.close()

        # Reset environment
        sim.reset()
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([0.0, 1.0, 0.0])
        agent.set_state(agent_state)
        goal_position = generate_random_goal(sim)
        print(f"Generated new goal position: {goal_position}")

# Main program entry
if __name__ == "__main__":
    train()
