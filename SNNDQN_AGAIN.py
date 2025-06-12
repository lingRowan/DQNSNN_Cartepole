 
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
from torchvision.transforms import ToPILImage



############ HYPERPARAMETERS ##############

BATCH_SIZE = 128 # original = 128
GAMMA = 0.999 # original = 0.999
EPS_START = 0.9 # original = 0.9
EPS_END = 0.05 # original = 0.05
EPS_DECAY = 1000 # original = 200
TARGET_UPDATE = 50 # original = 10
MEMORY_SIZE = 10000 # original = 10000
END_SCORE = 200 # 200 for Cartpole-v0
TRAINING_STOP = 142 # threshold for training stop
N_EPISODES = 500 # total episodes to be run
LAST_EPISODES_NUM = 20 # number of episodes for stopping training
FRAMES = 2 # state is the number of last frames: the more frames, 
# the more the state is detailed (still Markovian)
RESIZE_PIXELS = 60 # Downsample image to this number of pixels

# ---- CONVOLUTIONAL NEURAL NETWORK ----
HIDDEN_LAYER_1 = 16
HIDDEN_LAYER_2 = 32 
HIDDEN_LAYER_3 = 32
KERNEL_SIZE = 5 # original = 5
STRIDE = 2 # original = 2
# --------------------------------------

GRAYSCALE = True # False is RGB
LOAD_MODEL = False # If we want to load the model, Default= False
USE_CUDA = False # If we want to use GPU (powerful one needed!)
############################################

graph_name = 'Cartpole_Vision_Stop-' + str(TRAINING_STOP) + '_LastEpNum-' + str(LAST_EPISODES_NUM)
from torchvision import transforms

# Define your Gabor kernels and filters
kernels = [
    GaborKernel(window_size=3, orientation=45 + 22.5),
    GaborKernel(3, 90 + 22.5),
    GaborKernel(3, 135 + 22.5),
    GaborKernel(3, 180 + 22.5)
]
filt = Filter(kernels, use_abs=True)

def time_dim(input):
    return input.unsqueeze(0)  # This adds a dimension at the front for processing sequences

# Prepare the transformations
if GRAYSCALE == 0:
    transform = transforms.Compose([
        transforms.Resize((RESIZE_PIXELS, RESIZE_PIXELS), interpolation=Image.BICUBIC),
        transforms.ToTensor(),  # Convert image to tensor of shape (C, H, W)
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Example of repeating, if you want RGB
        time_dim,
        filt,
        pointwise_inhibition,
        Intensity2Latency(number_of_spike_bins=15, to_spike=True)
    ])
    nn_inputs = 3 * FRAMES  # This should match number of channels you're working with

else:
    transform = transforms.Compose([
        transforms.Resize((RESIZE_PIXELS, RESIZE_PIXELS), interpolation=Image.BICUBIC),
        transforms.Grayscale(num_output_channels=1),  # Use 1 channel for grayscale
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.repeat(4, 1, 1)),  
        filt,
        pointwise_inhibition,
        Intensity2Latency(number_of_spike_bins=15, to_spike=True)
    ])
    nn_inputs = 4 * FRAMES  
             
stop_training = False 

env = gym.make('CartPole-v1', render_mode = 'rgb_array').unwrapped

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# If gpu is to be used
device = torch.device("cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Memory for Experience Replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None) # if we haven't reached full capacity, we append a new transition
        self.memory[self.position] = Transition(*args)  
        self.position = (self.position + 1) % self.capacity # e.g if the capacity is 100, and our position is now 101, we don't append to
        # position 101 (impossible), but to position 1 (its remainder), overwriting old data

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) 

    def __len__(self): 
        return len(self.memory)

# Build CNN
class DQN(nn.Module):

    def __init__(self, input_channels, features_per_class, number_of_classes):
            super(DQN, self).__init__()
            self.features_per_class = features_per_class
            self.number_of_classes = number_of_classes
            self.number_of_features = features_per_class * number_of_classes
            self.pool = Pooling(kernel_size=5, stride=1, padding=1)
            #self.conv = nn.Conv2d(in_channels=8, out_channels=40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv = Convolution(input_channels, self.number_of_features, 3, 0.8, 0.05)
            #self.conv.reset_weight()
            self.stdp = STDP(conv_layer=self.conv, learning_rate=(0.05, -0.0005), use_stabilizer = True, lower_bound = 0, upper_bound = 1)
            self.anti_stdp = STDP(conv_layer=self.conv, learning_rate=(-0.05, 0.0005),use_stabilizer = True, lower_bound = 0, upper_bound = 1)
        
        # internal state of the model
            self.ctx = {"input_spikes": None, "potentials": None, "output_spikes": None, "winners": None}
        
        # map each neuron to the class it represents
            self.decision_map = []
            for i in range(number_of_classes):
                self.decision_map.extend([i] * features_per_class)

    def forward(self, x):
        # Proceed with the transformed input
        x = self.pool(x)
        p = self.conv(x)

        spikes, potentials = fire(potentials=p, threshold=20, return_thresholded_potentials=True)
        winners = get_k_winners(potentials=p, kwta=1, inhibition_radius=0, spikes=spikes)

        self.ctx["input_spikes"] = x
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = spikes
        self.ctx["winners"] = winners

        output = -1
        #output = torch.zeros(x.size(0), self.number_of_classes, device=x.device)
        if len(winners) != 0:
            output = self.decision_map[0]
        return output
        
    def get_weights(self):
            weight_mean =  np.mean(self.conv.weights.detach().cpu().numpy())
            return weight_mean


    def SNN_reward(self, reward):
            if reward > 0:
                print('True', reward.item())
                self.stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

            else:
                print('False', reward.item())
                self.anti_stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

# Cart location for centering image crop
def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

# Cropping, downsampling (and Grayscaling) image
'''def get_screen():
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
    return image'''

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

eps_threshold = 0.9 # original = 0.9  # exploration or exploitation

init_screen = get_screen()
_, screen_height, screen_width = np.array(init_screen).shape
print("Screen height: ", screen_height," | Width: ", screen_width)

# Get number of actions from gym action space
n_actions = env.action_space.n
model = DQN(360,20,2).to(device)
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if LOAD_MODEL == True: # getting the weight of the network from the files
    policy_net_checkpoint = torch.load('save_model/policy_net_best3.pt') # best 3 is the default best
    target_net_checkpoint = torch.load('save_model/target_net_best3.pt')
    policy_net.load_state_dict(policy_net_checkpoint)  # acivate the weight dictionary
    target_net.load_state_dict(target_net_checkpoint)
    policy_net.eval()
    target_net.eval() # evaluate each network and it's parameters
    stop_training = True # if we want to load, then we don't train the network anymore

optimizer = optim.RMSprop(policy_net.parameters()) 
memory = ReplayMemory(MEMORY_SIZE) # save the previous experience

steps_done = 0

# Action selection , if stop training == True, only exploitation
def select_action(state, stop_training): #epsilon greedy policy 
    global steps_done
    sample = random.random() # get the random number to compare it with the threshold
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1 # counting number of steps
    # print('Epsilon = ', eps_threshold, end='\n')
    if sample > eps_threshold or stop_training: # compare the random number (sample) with the threshold
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            SNN_action = torch.tensor(model.forward(state))
            return SNN_action  # sample is bigger then return the action with the maximum Q-value
    else:
        return torch.tensor(random.randrange(n_actions))
# if sample is less or equal then return a random action

episode_durations = []


# Plotting
def plot_durations(score):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float) # time taken by each episode 
    episode_number = len(durations_t) 
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), label= 'Score')
    matplotlib.pyplot.hlines(195, 0, episode_number, colors='red', linestyles=':', label='Win Threshold') # minimum score required to consider an episode a success
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        last100_mean = means[episode_number -100].item()
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label= 'Last 100 mean')
        print('Episode: ', episode_number, ' | Score: ', score, '| Last 100 mean = ', last100_mean)
    plt.legend(loc='upper left')
    #plt.savefig('./save_graph/cartpole_dqn_vision_test.png') # for saving graph with latest 100 mean
    plt.pause(0.001)  # pause a bit so that plots are updated
    #plt.savefig('save_graph/' + graph_name)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
to_tensor = transforms.ToTensor()

    # Convert images to tensors if necessary
def convert_to_tensor(data):
        if isinstance(data, Image.Image):  # Check if the data is a PIL Image
            return to_tensor(data).unsqueeze(0)  # Convert to tensor and add batch dimension
        elif isinstance(data, torch.Tensor):  # Already a tensor
            return data
        else:
            raise TypeError(f"Expected input to be PIL Image or Tensor, but got {type(data)}")

def optimize_model():   
    if len(memory) < BATCH_SIZE:
        return

    # Sample a batch from memory
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Convert batch elements to tensors
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([convert_to_tensor(s) for s in batch.next_state if s is not None])
    
    state_batch = torch.cat([convert_to_tensor(s) for s in batch.state])
    action_batch = torch.cat([action.view(1) for action in batch.action]).view(-1)  # Ensure shape is (BATCH_SIZE,)
    
    # Convert rewards batch to tensor
    reward_batch = torch.cat(batch.reward)

    # Get the Q-values from the policy network
    state_action_values = torch.tensor(policy_net(state_batch))
    
   # if isinstance(state_action_values, torch.Tensor):
    #    state_action_values = state_action_values.gather(1, action_batch.view(-1, 1))
    #else:
     #   raise ValueError("Expected state_action_values to be a tensor.")

    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    next_state_values = target_net(non_final_next_states)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Ensure that loss has .requires_grad set to True
    #if not loss.requires_grad:
     #   raise ValueError("Loss does not require gradients.")

    # Optimize the model
    optimizer.zero_grad()
    #loss.backward()  # compute gradients
    
    # Clip the gradients
    #for param in policy_net.parameters():
     #   param.grad.data.clamp_(-1, 1)
    
    optimizer.step()
    
    # Clip the gradients
    '''for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    
    optimizer.step()'''
    
# Training 
'''def optimize_model(reward):   #visualizing the training process
    if reward > 0.5:
            print('True', reward)
            policy_net.stdp(policy_net.ctx["input_spikes"], policy_net.ctx["potentials"], policy_net.ctx["output_spikes"], policy_net.ctx["winners"])

    else:
            print('False', reward)
            policy_net.anti_stdp(policy_net.ctx["input_spikes"], policy_net.ctx["potentials"], policy_net.ctx["output_spikes"], policy_net.ctx["winners"])


    # Compute Huber loss
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #plt.figure(2)

    # Optimize the model
    #optimizer.zero_grad()
    #loss.backward()
    #for param in policy_net.parameters():
     #   param.grad.data.clamp_(-1, 1)
    #optimizer.step()'''

mean_last = deque([0] * LAST_EPISODES_NUM, LAST_EPISODES_NUM)
           

episode_data_list = []  # Initialize a list to collect episode data

# MAIN LOOP
'''image_directory = "saved_images"  
os.makedirs(image_directory, exist_ok=True)  
save_directory = 'save_model'
# Create the save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)'''

for i_episode in range(N_EPISODES):
    # Initialize the environment and state
    env.reset()
    init_screen = get_screen()
   
    screens = deque([transform(init_screen)] * FRAMES, FRAMES)
    state = torch.cat(list(screens), dim=1)
    #print("State shape:", state.shape) 
    #state = init_screen
    
    

    #screens = deque([transform(init_screen)] * FRAMES, FRAMES)
    
    for t in count():
        # Select and perform an action
        #state = join_image(*list(screens))  
        #action = select_action(state, stop_training)
        #action =policy_net(state)
        action = select_action(state, stop_training)
        #action = torch.tensor(model.forward(state))
        print('Action', action.item()) 
        state_variables, _, done, _, _ = env.step(action.item())
        

        # Observe new state
        #new_image = get_screen()
        #screens.append(new_image)
        #next_state = join_image(*list(screens)) 
        screens.append(transform(get_screen()))
        next_state = torch.cat(list(screens), dim=1) if not done else None 
        #next_state =  get_screen()
        #screens.append(transform(get_screen()))
        position, velocity, angle, angular_velocity = state_variables
        r1 = (env.x_threshold - abs(position)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(angle)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        reward = torch.tensor([reward], device=device)
        #model.SNN_reward(reward)        
        if t >= END_SCORE - 1:
            reward += 20
            done = True
        elif done:
            reward -= 20 

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Break out of the loop if done
        if done:
            episode_durations.append(t + 1)
            plot_durations(t + 1)
            mean_last.append(t + 1)
            mean = sum(mean_last) / LAST_EPISODES_NUM

            if mean < TRAINING_STOP and not stop_training:
                optimize_model()
            else:
                stop_training = True

            break  # Exit the loop once the episode is done

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:  
        target_net.load_state_dict(policy_net.state_dict())

# Save the pretrained model after training is complete
#torch.save(policy_net.state_dict(), 'save_model/policy_net_final.pt')
#torch.save(target_net.state_dict(), 'save_model/target_net_final.pt')

print('Model saved successfully.')
print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()


