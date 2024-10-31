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

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)  
        self.position = (self.position + 1) % self.capacity 

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) 

    def __len__(self): 
        return len(self.memory)

class SNN(nn.Module):
        def __init__(self, input_channels, features_per_class, number_of_classes):
            super(SNN, self).__init__()
            self.features_per_class = features_per_class
            self.number_of_classes = number_of_classes
            self.number_of_features = features_per_class * number_of_classes
            self.pool = Pooling(kernel_size=5, stride=1, padding=1)
            self.conv = Convolution(input_channels, self.number_of_features, 3, 0.8, 0.05)
            self.conv.reset_weight()
            self.stdp = STDP(conv_layer=self.conv, learning_rate=(0.05, -0.015), use_stabilizer = True, lower_bound = 0, upper_bound = 1)
            self.anti_stdp = STDP(conv_layer=self.conv, learning_rate=(-0.05, 0.0005),use_stabilizer = True, lower_bound = 0, upper_bound = 1)
        
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


        def reward(self, reward):
            if reward.item() >= 0:  
                print('Positive Reward', reward.item())
                self.stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

            elif reward.item() < 0:  
                print('Severe Failure', reward.item())
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
   
def join_image(img1, img2):
    img = Image.new('L', (img1.width + img2.width, img1.height))
    img.paste(img1, (0, 0))
    img.paste(img2, (img1.width, 0))
    return img

env.reset()
'''plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), cmap='gray')
plt.title('Example extracted screen')
plt.show()'''

eps_threshold = 0.9

init_screen = get_screen()
_, screen_height, screen_width = np.array(init_screen).shape
print("Screen height: ", screen_height, " | Width: ", screen_width)

# Get number of actions from gym action space
n_actions = env.action_space.n

# Initialize policy and target networks
model = SNN(4, 50, 2)

memory = ReplayMemory(MEMORY_SIZE)

def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    action = model.forward(state)
    
    if random.random() > eps_threshold:
        return action  
    else:
        print('RANDOM')
        return random.randrange(n_actions)



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

# MAIN LOOP

for i_episode in range(N_EPISODES):
    env.reset()
    init_screen = get_screen()  
    screens = deque([init_screen] * FRAMES, maxlen=FRAMES)
    '''if i_episode > 0 and i_episode % 50 == 0:  
      model.conv.reset_weight()'''
    
    for t in count():
        
        state = join_image(*list(screens))  
        #action = model.forward(state)
        action = model.forward(state)
        print('Action ', action)
        weights = model.get_weights()
        #print('weight', weights)
        state_variables, _, done, _, _= env.step(action)
        new_image = get_screen()
        screens.append(new_image)
        next_state = join_image(*list(screens))  
        position, velocity, angle, angular_velocity = state_variables
        '''r1 = (env.x_threshold - abs(position)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(angle)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        reward = torch.tensor([reward], device=device)

        
        model.reward(reward)
        #print('angle', env.theta_threshold_radians)

        if t >= END_SCORE - 1:
            reward += 20
            done = True
        elif done:
            reward -= 20 '''
        # Get thresholds
        theta_threshold = env.theta_threshold_radians
        x_threshold = env.x_threshold
        angle_reward = 0.5 - (abs(angle) / theta_threshold)  
        position_reward = 0.5 - (abs(position) / x_threshold)  
        reward = angle_reward + position_reward
        if abs(position) > (0.5 * x_threshold):
            reward -= 0.5  
        if abs(angle) > (0.5 * theta_threshold):
            reward -= 0.5  
        if abs(angle) >= theta_threshold or abs(position) >= x_threshold:
            reward = -10.0  
        reward = torch.tensor([reward], device=device)

        model.reward(reward)
        '''if t >= END_SCORE - 1:
            reward += 20
            done = True
        elif done:
            reward -= 20 '''

        memory.push(state, action, next_state, reward)
        state = next_state
        
        if done:
            episode_durations.append(t + 1)
            plot_durations(t + 1)
            mean_last.append(t + 1)
            mean = sum(mean_last) / LAST_EPISODES_NUM
            
            if mean < TRAINING_STOP and not stop_training:
                model.reward(reward)
            else:
                stop_training = True

            break

'''for i_episode in range(N_EPISODES):
    env.reset()
    init_screen = get_screen()
    sec_screen = get_screen()
    state = join_image(init_screen, sec_screen)
    print('State', state)
    
    for t in count():
        action = model.forward(state)
        print('Action ', action)
        state_variables, _, done, _, _ = env.step(action)
        img = get_screen()
        img2 = get_screen()
        next_state = join_image(img, img2)
        position, velocity, angle, angular_velocity = state_variables
        r1 = (env.x_threshold - abs(position)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(angle)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        reward = torch.tensor([reward], device=device)
        model.reward(reward)
    
       

        memory.push(state, action, next_state, reward)
        state = next_state

        if done:
            episode_durations.append(t + 1)
            plot_durations(t + 1)
            mean_last.append(t + 1)
            mean = sum(mean_last) / LAST_EPISODES_NUM

            if mean < TRAINING_STOP and not stop_training:
                model.reward(reward)
            else:
                stop_training = True

            break'''

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()