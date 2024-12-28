import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from collections import deque
import os
from snn import *
from functional import *
from utils import *
from torchvision import transforms
from PIL import Image

############ HYPERPARAMETERS ##############
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 50
MEMORY_SIZE = 10000
END_SCORE = 200
TRAINING_STOP = 142
N_EPISODES = 500
LAST_EPISODES_NUM = 20
FRAMES = 2
RESIZE_PIXELS = 60

# ---- CONVOLUTIONAL NEURAL NETWORK ----
HIDDEN_LAYER_1 = 16
HIDDEN_LAYER_2 = 32
HIDDEN_LAYER_3 = 32
KERNEL_SIZE = 5
STRIDE = 2
# --------------------------------------

GRAYSCALE = True
LOAD_MODEL = False
USE_CUDA = False
############################################
kernels = [
    GaborKernel(window_size=3, orientation=45 + 22.5),
    GaborKernel(3, 90 + 22.5),
    GaborKernel(3, 135 + 22.5),
    GaborKernel(3, 180 + 22.5)]
filt = Filter(kernels, use_abs=True)

def time_dim(input):
    return input.unsqueeze(0)

# Transformation Pipeline
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    time_dim,
    filt,
    pointwise_inhibition,
    Intensity2Latency(number_of_spike_bins=15, to_spike=True)
])

graph_name = 'Cartpole_Vision_Stop-' + str(TRAINING_STOP) + '_LastEpNum-' + str(LAST_EPISODES_NUM)

stop_training = False 
env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Define SNN Class
class SNN(nn.Module):
    def __init__(self, input_channels, features_per_class, number_of_classes):
        super(SNN, self).__init__()
        self.features_per_class = features_per_class
        self.number_of_classes = number_of_classes
        self.number_of_features = features_per_class * number_of_classes
        self.pool = Pooling(kernel_size=5, stride=1, padding=1)
        self.conv = Convolution(input_channels, self.number_of_features, 3, 0.8, 0.05)
        self.stdp = STDP(conv_layer=self.conv, learning_rate=(0.005, -0.15), use_stabilizer=True, lower_bound=0, upper_bound=1)
        self.anti_stdp = STDP(conv_layer=self.conv, learning_rate=(-0.05, 0.0005), use_stabilizer=True, lower_bound=0, upper_bound=1)
        
        # Internal state of the model
        self.ctx = {"input_spikes": None, "potentials": None, "output_spikes": None, "winners": None}
        
        # Map each neuron to the class it represents
        self.decision_map = []
        for i in range(number_of_classes):
            self.decision_map.extend([i] * features_per_class)

    def forward(self, x):
        x = self.pool(x)
        p = self.conv(x)
        spikes, potentials = fire(potentials=p, threshold=20, return_thresholded_potentials=True)
        winners = get_k_winners(potentials=p, kwta=1, inhibition_radius=0, spikes=spikes)
        self.ctx["input_spikes"] = x
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = spikes
        self.ctx["winners"] = winners

        output = -1
        if len(winners) != 0:
            output = self.decision_map[winners[0][0]]
        return output

    def reward(self):
        self.stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        print('Reward')

    def punish(self):
        self.anti_stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        print('Punishment')

# Helper functions
def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

def get_screen():
    screen = env.render().transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen.transpose(1, 2, 0), dtype=np.uint8)
    image = Image.fromarray(screen)
    return transform(image)

# Initialize the environment
env.reset()
eps_threshold = 0.9

init_screen = get_screen()
_, _, screen_height, screen_width = np.array(init_screen).shape

# Get number of actions from gym action space
n_actions = env.action_space.n

# Initialize policy and target networks
model = SNN(4, 20, 2)

steps_done = 0
episode_accuracies = []

# New plotting function for accuracy
def plot_accuracy(accuracy_scores):
    plt.figure(2)
    plt.clf()
    episode_number = len(accuracy_scores)

    plt.title('Training Accuracy over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.plot(range(episode_number), accuracy_scores, label='Accuracy', color='blue')
    plt.hlines(1.0, 0, episode_number, colors='red', linestyles=':', label='Perfect Accuracy')
    plt.legend(loc='upper left')
    plt.grid()
    plt.pause(0.001)

# MAIN LOOP
for i_episode in range(N_EPISODES):
    env.reset()
    init_screen = get_screen()
    screens = deque([init_screen] * FRAMES, FRAMES)

    total_reward = 0
    num_actions = 0
    
    for t in count():
        state = init_screen
        action = model.forward(state)
        print('Action: ', action)
        
        # Take action in the environment
        state_variables, reward, done, _, _ = env.step(action)
        screens.append(get_screen())
        next_state = torch.cat(list(screens), dim=1) if not done else None

        # Count actions
        num_actions += 1

        position, velocity, angle, angular_velocity = state_variables
        print('Angle:', angle)
        print('Position: ', position)
        reward_received = False


        if position > 0 and angle > 0 and action == 1:
            model.reward()
            reward_received = True
        elif position < 0 and angle < 0 and action == 0:
            model.reward()
            reward_received = True
        elif (position > 0 and angle < 0 and action == 1) or (position < 0 and angle > 0 and action == 0):
            model.reward()
            reward_received = True
        else:
            model.punish()

        # Check if a reward was received to count as a correct action
        if reward_received:
            total_reward += 1

        state = next_state
        
        if done:
            # Calculate accuracy as the number of rewards over total actions
            accuracy = total_reward / num_actions if num_actions > 0 else 0
            print()
            print('Accuracy: ', accuracy)
            print()
            episode_accuracies.append(accuracy)

            plot_accuracy(episode_accuracies)
            break

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
