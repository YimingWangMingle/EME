import torch
from torch import nn
from torch.nn import functional as F


class mlp_net(nn.Module):
    def __init__(self, state_size, num_actions, dist_type):
        super(mlp_net, self).__init__()
        self.dist_type = dist_type
        self.fc1_v = nn.Linear(state_size, 64)
        self.fc2_v = nn.Linear(64, 64)
        self.fc1_a = nn.Linear(state_size, 64)
        self.fc2_a = nn.Linear(64, 64)
        # check the type of distribution
        if self.dist_type == 'gauss':
            self.sigma_log = nn.Parameter(torch.zeros(1, num_actions))
            self.action_mean = nn.Linear(64, num_actions)
            self.action_mean.weight.data.mul_(0.1)
            self.action_mean.bias.data.zero_()
        elif self.dist_type == 'beta':
            self.action_alpha = nn.Linear(64, num_actions)
            self.action_beta = nn.Linear(64, num_actions)
            # init..
            self.action_alpha.weight.data.mul_(0.1)
            self.action_alpha.bias.data.zero_()
            self.action_beta.weight.data.mul_(0.1)
            self.action_beta.bias.data.zero_()

        # define layers to output state value
        self.value = nn.Linear(64, 1)
        self.value.weight.data.mul_(0.1)
        self.value.bias.data.zero_()

    def forward(self, x):
        x_v = torch.tanh(self.fc1_v(x))
        x_v = torch.tanh(self.fc2_v(x_v))
        state_value = self.value(x_v)
        # output the policy...
        x_a = torch.tanh(self.fc1_a(x))
        x_a = torch.tanh(self.fc2_a(x_a))
        if self.dist_type == 'gauss':
            mean = self.action_mean(x_a)
            sigma_log = self.sigma_log.expand_as(mean)
            sigma = torch.exp(sigma_log)
            pi = (mean, sigma)
        elif self.dist_type == 'beta':
            alpha = F.softplus(self.action_alpha(x_a)) + 1
            beta = F.softplus(self.action_beta(x_a)) + 1
            pi = (alpha, beta)

        return state_value, pi


class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 512) 
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return x


class cnn_net(nn.Module):
    def __init__(self, num_actions):
        super(cnn_net, self).__init__()
        self.cnn_layer = deepmind()
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        value = self.critic(x)
        pi = F.softmax(self.actor(x), dim=1)
        return value, pi


class mlp_intrinsic_net(nn.Module):
    
    def __init__(self, state_size, num_actions):
        super(mlp_intrinsic_net, self).__init__()
        self.fc1_r_in = nn.Linear(state_size + num_actions, 64)
        self.fc2_r_in = nn.Linear(64, 64)
        self.r_in = nn.Linear(64, 1)
        self.fc1_v_ex = nn.Linear(state_size, 64)
        self.fc2_v_ex = nn.Linear(64, 64)
        self.v_ex = nn.Linear(64, 1)
        self.r_in.weight.data.mul_(0.1)
        self.r_in.bias.data.zero_()
        self.v_ex.weight.data.mul_(0.1)
        self.v_ex.bias.data.zero_()
    
    def forward(self, x, actions):
        if actions is not None:
            r_in_inputs = torch.cat([x, actions], dim=1)
            x_a = F.relu(self.fc1_r_in(r_in_inputs))
            x_a = F.relu(self.fc2_r_in(x_a))
            r_in = torch.tanh(self.r_in(x_a))
        else:
            r_in = None
        x_v_ex = F.relu(self.fc1_v_ex(x))
        x_v_ex = F.relu(self.fc2_v_ex(x_v_ex))
        v_ex = self.v_ex(x_v_ex)
        return r_in, v_ex


class mlp_inv_net(nn.Module):

    def __init__(self, state_size):
        super(mlp_inv_net, self).__init__()
        self.fc1_feat_in = nn.Linear(state_size, 64)
        self.fc2_feat_in = nn.Linear(64, 64)
        self.feat_in = nn.Linear(64, 64)
        self.fc1_v_ex = nn.Linear(state_size, 64)
        self.fc2_v_ex = nn.Linear(64, 64)
        self.v_ex = nn.Linear(64, 1)
        self.v_ex.weight.data.mul_(0.1)
        self.v_ex.bias.data.zero_()

    def forward(self, x):
        x_feat_in = F.relu(self.fc1_feat_in(x))
        x_feat_in = F.relu(self.fc2_feat_in(x_feat_in))
        feat_in = self.feat_in(x_feat_in)
        x_v_ex = F.relu(self.fc1_v_ex(x))
        x_v_ex = F.relu(self.fc2_v_ex(x_v_ex))
        v_ex = self.v_ex(x_v_ex)
        return feat_in, v_ex
