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
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque
import os
from snn import *
from functional import *
from utils import *
from torchvision import transforms
#from torchvision.transforms import ToPILImage
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
kernels = [	GaborKernel(window_size = 3, orientation = 45+22.5),
            GaborKernel(3, 90+22.5),
            GaborKernel(3, 135+22.5),
            GaborKernel(3, 180+22.5)]
filt = Filter(kernels, use_abs = True)

def time_dim(input):
    return input.unsqueeze(0)

transform = transforms.Compose(
    [transforms.Grayscale(),
    transforms.ToTensor(),
    time_dim,
    filt,
    pointwise_inhibition,
    Intensity2Latency(number_of_spike_bins = 15, to_spike = True)])

graph_name = 'Cartpole_Vision_Stop-' + str(TRAINING_STOP) + '_LastEpNum-' + str(LAST_EPISODES_NUM)

if GRAYSCALE == 0:
    resize = T.Compose([T.ToPILImage(), 
                    T.Resize(RESIZE_PIXELS, interpolation=Image.BICUBIC),
                    T.ToTensor()])
    
    nn_inputs = 3*FRAMES  # number of channels for the nn
else:
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(RESIZE_PIXELS, interpolation=Image.BICUBIC),
                    T.Grayscale(),
                    T.ToTensor()])
    nn_inputs =  FRAMES # number of channels for the nn

                    
stop_training = False 

env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class SNN(nn.Module):
        def __init__(self, input_channels, features_per_class, number_of_classes, stdp_lr, anti_stdp_lr):
            super(SNN, self).__init__()
            self.features_per_class = features_per_class
            self.number_of_classes = number_of_classes
            self.number_of_features = features_per_class * number_of_classes
            self.pool = Pooling(kernel_size=5, stride=1, padding=1)
            self.conv = Convolution(input_channels, self.number_of_features, 3, 0.8, 0.05)
            self.conv.reset_weight()
            self.stdp_lr = stdp_lr
            self.anti_stdp_lr = anti_stdp_lr
            self.stdp = STDP(self.conv , stdp_lr)
            self.anti_stdp = STDP(self.conv , anti_stdp_lr)
            #self.stdp = STDP(conv_layer=self.conv, learning_rate=(0.09, -0.5), use_stabilizer = True, lower_bound = 0, upper_bound = 1)
            #self.anti_stdp = STDP(conv_layer=self.conv, learning_rate=(-0.5, 0.05),use_stabilizer = True, lower_bound = 0, upper_bound = 1)
        
        # internal state of the model
            self.ctx = {"input_spikes": None, "potentials": None, "output_spikes": None, "winners": None}
        
        # map each neuron to the class it represents
            self.decision_map = []
            for i in range(number_of_classes):
                self.decision_map.extend([i] * features_per_class)

    
        def forward(self, x):
            x = transform(x)
            x = self.pool(x)
            p = self.conv(x)
            spikes, potentials = fire(potentials=p, threshold=20, return_thresholded_potentials=True)
            #print('potenials', potentials)
            winners = get_k_winners(potentials=p, kwta=1, inhibition_radius=0, spikes=spikes)
            self.ctx["input_spikes"] = x
            self.ctx["potentials"] = potentials
            self.ctx["output_spikes"] = spikes
            self.ctx["winners"] = winners
            #print('winner', winners)

            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
        
        def get_weights(self):
            weight_mean =  np.mean(self.conv.weights.detach().cpu().numpy())
            return weight_mean
        
        def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
            self.stdp.update_all_learning_rate(stdp_ap, stdp_an)
            self.anti_stdp.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)


        def reward(self, reward):
            if reward > 0:
                print('True', reward.item())
                print()
                self.stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

            else:
                print('False', reward.item())
                print()
                self.anti_stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
           
       
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
    if screen.shape[2] == 3:
        image = Image.fromarray(screen)
    else:
        image = Image.fromarray(screen[:, :, 0]) 

    return image

env.reset()

eps_threshold = 0.9

init_screen = get_screen()
_, screen_height, screen_width = np.array(init_screen).shape
print("Screen height: ", screen_height, " | Width: ", screen_width)

# Get number of actions from gym action space
n_actions = env.action_space.n

# Initialize policy and target networks
model = SNN(4, 20, 2, (0.05, -0.015), (-0.05, 0.0005))

steps_done = 0

episode_durations = []

def plot_durations(score):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    episode_number = len(durations_t) 
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), label='Score')
    matplotlib.pyplot.hlines(195, 0, episode_number, colors='red', linestyles=':', label='Win Threshold')
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        last100_mean = means[episode_number - 100].item()
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='Last 100 mean')
        print('Episode: ', episode_number, ' | Score: ', score, '| Last 100 mean = ', last100_mean)
    plt.legend(loc='upper left')
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    
mean_last = deque([0] * LAST_EPISODES_NUM, LAST_EPISODES_NUM)
# initial adaptive learning rates
apr = model.stdp_lr[0]
anr = model.stdp_lr[1]
app = model.anti_stdp_lr[1]
anp = model.anti_stdp_lr[0]

adaptive_min = 0.2
adaptive_int = 0.8
apr_adapt = ((1.0 - 1.0 / model.number_of_classes) * adaptive_int + adaptive_min) * apr
anr_adapt = ((1.0 - 1.0 / model.number_of_classes) * adaptive_int + adaptive_min) * anr
app_adapt = ((1.0 / model.number_of_classes) * adaptive_int + adaptive_min) * app
anp_adapt = ((1.0 / model.number_of_classes) * adaptive_int + adaptive_min) * anp
# MAIN LOOP

for i_episode in range(N_EPISODES):
    env.reset()
    init_screen = get_screen()  
    
    
    for t in count():
        model.train()
        state = init_screen
        action = model.forward(state)
        print('Action ', action)
        
        weights = model.get_weights()
        #print('weight', weights)
        state_variables, _, done, _, _= env.step(action)
        new_image = get_screen()
        next_state = new_image 
        position, velocity, angle, angular_velocity = state_variables
        r1 = (env.x_threshold - abs(position)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(angle)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        reward = torch.tensor([reward], device=device)
        model.reward(reward)
        #print('reward_after', reward.item())
        state = next_state
        
        if done:
            episode_durations.append(t + 1)
            plot_durations(t + 1)
            mean_last.append(t + 1)
            mean = sum(mean_last) / LAST_EPISODES_NUM
            
            if mean < TRAINING_STOP and not stop_training:
               # model.reward(reward)
                pass
            else:
                stop_training = True

            break

        
        #model.reward(reward)
        #print('angle', env.theta_threshold_radians)

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()