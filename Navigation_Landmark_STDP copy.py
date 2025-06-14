import gym
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
from snn import *
from snn import Convolution
from functional import *
from utils import *
from torchvision import transforms
from PIL import Image
from gym import spaces

############ HYPERPARAMETERS ##############
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 50
MEMORY_SIZE = 500
END_SCORE = 200
#TRAINING_STOP = 142
N_EPISODES = 1000
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
    Intensity2Latency(number_of_spike_bins=4, to_spike=True)
])

class CustomNavigationEnv(gym.Env):
    def __init__(self):
        super(CustomNavigationEnv, self).__init__()

        # Define action space: 4 possible actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        
        # Initialize agent and goal positions
        self.goal_position = (10, 10)  # Goal position (fixed for simplicity)
        self.current_position = (0, 0)  # Starting position
        self.previous_position = (0, 0)  # Track previous position for reward calculation

        # Define the grid size
        self.grid_size = 12

        # Creating random block positions (walls)
        self.blocks = self.generate_random_blocks()
        
        # Generate tree positions (near the goal)
        self.trees = self.generate_tree_positions()  # Generate tree positions adjacent to the goal
        
        # Create figure for visualization
        self.fig, self.ax = plt.subplots()

        # Initialize the state
        self.state = self.get_state()

    def generate_random_blocks(self, num_blocks=4, min_distance=2):
        """ Generate random block positions ensuring they are at least 'min_distance' apart. """
        blocks = set()
    
        while len(blocks) < num_blocks:
            block_position = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
        
        # Ensure block is not at the goal, start position, and is far enough from other blocks
            if block_position != self.goal_position and block_position != self.current_position:
            # Check if the new block is at least 'min_distance' away from all existing blocks
                if all(np.linalg.norm(np.array(block_position) - np.array(b)) >= min_distance for b in blocks):
                   blocks.add(block_position)
    
        return blocks
    
    def reset(self):
        # Reset the agent's position to start at (0, 0)
        self.current_position = (0, 0)
        self.previous_position = (0, 0)
        
        # Re-generate random blocks each time we reset
        self.blocks = self.generate_random_blocks()

        # Update the state
        self.state = self.get_state()  # Reset the state to the initial grid
        return self.state
    def generate_tree_positions(self):
    
     adjacent_positions = [
        (self.goal_position[0] - 3, self.goal_position[1]),  # Up
        (self.goal_position[0] + 3, self.goal_position[1]),  # Down
        (self.goal_position[0], self.goal_position[1] - 3),  # Left
        (self.goal_position[0], self.goal_position[1] + 3)   # Right
    ]
    
    # Remove any out-of-bounds positions
     valid_trees = []
     for pos in adjacent_positions:
        if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
            valid_trees.append(pos)
     return valid_trees[:2]  

    def step(self, action):
    # Store the previous position
     self.previous_position = self.current_position
    
    # Calculate the next position based on the action
     next_position = list(self.current_position)
    
     if action == 0 and next_position[0] > 0:  # Move up
        next_position[0] -= 1
     elif action == 1 and next_position[0] < self.grid_size - 1:  # Move down
        next_position[0] += 1
     elif action == 2 and next_position[1] > 0:  # Move left
        next_position[1] -= 1
     elif action == 3 and next_position[1] < self.grid_size - 1:  # Move right
        next_position[1] += 1

     current_reward = 0  # Default reward

    # Check if the next position hits a block (wall)
     if tuple(next_position) in self.blocks:
        current_reward = -2
        self.current_position = self.previous_position  # Stay in the previous position
     else:
        # Update position only if not hitting a block
        self.current_position = tuple(next_position)
        current_distance = np.linalg.norm(np.array(self.current_position) - np.array(self.goal_position))
        previous_distance = np.linalg.norm(np.array(self.previous_position) - np.array(self.goal_position))
        if current_distance < previous_distance:
            current_reward = 1
        elif current_distance > previous_distance:
            current_reward = -1
        elif current_distance > previous_distance:
                current_reward = -1

        # Give reward if reached a tree
        if tuple(self.current_position) in self.trees:
            current_reward = 2  # Reward for reaching a tree

        # Logic for reaching the goal
        if self.current_position == self.goal_position:
            current_reward = 5 # Larger reward for reaching the goal

     done = self.current_position == self.goal_position

    # Update the state
     self.state = self.get_state()  # Get the updated state (grid)

    # Return the new state, reward, done flag, and additional info
     return self.state, current_reward, done, {}


    def render(self):
     self.ax.clear()
    
    # Set grid limits
     self.ax.set_xlim(-0.5, self.grid_size - 0.5)
     self.ax.set_ylim(-0.5, self.grid_size - 0.5)

    # Draw the grid for walls
     self.ax.set_xticks(np.arange(0, self.grid_size, 1))
     self.ax.set_yticks(np.arange(0, self.grid_size, 1))
     self.ax.grid(True)

    # Plot the walls (blocks)
     wall_width = 0.5
     for block in self.blocks:
        block_x, block_y = block
        self.ax.add_patch(plt.Rectangle((block_y - wall_width / 2, block_x - wall_width / 2), 
                                          wall_width, wall_width, color='blue'))

    # Plot the robot (agent)
     robot_size = 0.4
     self.ax.add_patch(plt.Rectangle((self.current_position[1] - robot_size / 2, 
                                      self.current_position[0] - robot_size / 2), 
                                      robot_size, robot_size, color='red', label='Robot'))

    # Plot the goal's position
     goal_x, goal_y = self.goal_position
     self.ax.plot(goal_y, goal_x, 'go', markersize=10, label='Goal')

    # Plot the trees (as green rectangles)
     for tree in self.trees:
        tree_x, tree_y = tree
        self.ax.add_patch(plt.Rectangle((tree_y - 0.2, tree_x - 0.2), 0.4, 0.4, color='green', label='Tree'))

    # Add labels and title
     self.ax.set_xlabel('X Position')
     self.ax.set_ylabel('Y Position')
     self.ax.set_title('Agent Navigation')

    # Display the legend
     self.ax.legend()

    # Show the plot
     plt.pause(0.001)  # Pause for a brief moment to update the display

    def get_state(self):
    
    # Create a blank grid (e.g., 12x12 grid with RGB channels)
     screen = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)  # RGB grid

    # Mark the agent's position (robot)
     agent_x, agent_y = self.current_position
     screen[agent_x, agent_y] = [255, 0, 0]  # Color for the robot (red)

    # Mark the goal's position (green)
     goal_x, goal_y = self.goal_position
     screen[goal_x, goal_y] = [0, 255, 0]  # Green color for the goal

    # Mark the blocks (walls)
     for block in self.blocks:
        block_x, block_y = block
        screen[block_x, block_y] = [0, 0, 255]  # Wall color (blue)

    # Convert the screen (grid) to an image using PIL
     image = Image.fromarray(screen)

    # Apply transformations (e.g., resizing, converting to tensor)
     return transform(image)  # Return the transformed imagesor)
    
graph_name = 'CustomNavigationEnv_Stop-' + '_LastEpNum-' + str(LAST_EPISODES_NUM)

# Initialize the custom environment
env = CustomNavigationEnv()  # Replace with your custom environment

# Set up matplotlib for interactive plotting
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()  


# Define SNN Class
class SNN(nn.Module):
    def __init__(self, input_channels, features_per_class, number_of_classes):
        super(SNN, self).__init__()
        self.features_per_class = features_per_class
        self.number_of_classes = number_of_classes
        self.number_of_features = features_per_class * number_of_classes
        self.pool = Pooling(kernel_size=5, stride=1, padding=1)
        self.conv = Convolution(input_channels, self.number_of_features, 3, 0.8, 0.05)
        
        # Updated STDP and anti-STDP learning rates  
        self.stdp1 = STDP(conv_layer=self.conv, learning_rate=(0.009, -0.0001), use_stabilizer=True, lower_bound=0, upper_bound=1) # reach to the goal
        self.stdp2 = STDP(conv_layer=self.conv, learning_rate=(0.001, -0.00001), use_stabilizer=True, lower_bound=0, upper_bound=1) # closer to the goal
        self.stdp3 = STDP(conv_layer=self.conv, learning_rate=(0.005, -0.00005), use_stabilizer=True, lower_bound=0, upper_bound=1)  # landmark

        self.anti_stdp1 = STDP(conv_layer=self.conv, learning_rate=(-0.05, 0.00005), use_stabilizer=True, lower_bound=0, upper_bound=1)  # hit the block
        self.anti_stdp2 = STDP(conv_layer=self.conv, learning_rate=(-0.09, 0.0001), use_stabilizer=True, lower_bound=0, upper_bound=1) # far away from the goal
        self.anti_stdp3 = STDP(conv_layer=self.conv, learning_rate=(-0.01, 0.0003), use_stabilizer=True, lower_bound=0, upper_bound=1) # no change, no reward
        #self.normalize_weights()
        
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

    def reward(self, current_reward):
        if current_reward == 5:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
            print('Goal Reached')
        elif current_reward == 1:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
            print('You are close to the goal')
        elif current_reward == 2:
            self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
            print('You reached to the Landmark')


       

    def punish(self, current_reward):
        if current_reward == -1:
            self.anti_stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
            print('You are far away from the goal')
        elif current_reward == -2:
            self.anti_stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
            print('You hit the Block')
        elif current_reward == 0:
            self.anti_stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
            print('punishment')


def get_screen():
    """
    Captures the current visual representation of the environment and transforms it.
    """
    env.render()  # Render the environment state to the current plot
    
    # Obtain the state from the environment, which is the current visual grid
    image = env.get_state()  # Get the current state as an image
    
    # Assuming the 'transform' function defined earlier (including time dimension and filters)
    return image  # Transform the image as needed

# Initialize the environment
env.reset()
eps_threshold = 0.9

init_screen = get_screen()
_, _, screen_height, screen_width = np.array(init_screen).shape

# Get number of actions from gym action space
n_actions = env.action_space.n

# Initialize policy and target networks
model = SNN(8, 40, 4)

steps_done = 0
episode_accuracies = []

def moving_average(data, window_size=10):
    """Compute moving average over the last 'window_size' elements."""
    if len(data) < window_size:
        return np.mean(data)  # Use mean of available values if not enough data
    return np.mean(data[-window_size:])  # Compute moving average of last N elements

def plot_accuracy(accuracy_scores,  window_size=10):
    plt.figure(2)
    plt.clf()
    episode_number = len(accuracy_scores)
    smoothed_accuracies = [moving_average(accuracy_scores[:i+1], window_size) for i in range(episode_number)]

    #plt.title('Average Reward per action')
    plt.title('Cumulative Reward per Episode')
    plt.xlabel('Episode')
    #plt.ylabel('Average Rward')
    plt.ylabel('Cumulative Reward')
    plt.plot(range(episode_number), accuracy_scores, label='Accuracy', color='blue')
    plt.plot(range(episode_number), smoothed_accuracies, label=f'Moving Avg ({window_size} episodes)', color='red')
    plt.hlines(1.0, 0, episode_number, colors='red', linestyles=':', label='Perfect Accuracy')
    plt.legend(loc='upper left')
    plt.grid()
    plt.pause(0.001)

def plot_difference(difference, window_size=10):
    plt.figure(2)
    plt.clf()
    episode_number = len(difference)
    smoothed_accuracies = [moving_average(steps_per_episode[:i+1], window_size) for i in range(episode_number)]

    plt.title('Difference from optimal steps over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Difference')
    plt.plot(range(episode_number), difference, label='Differences from Optimal Steps', color='blue')
    plt.plot(range(episode_number), smoothed_accuracies, label=f'Moving Avg ({window_size} episodes)', color='red')
    plt.hlines(0, 0, episode_number, colors='red', linestyles=':', label='No difference')
    plt.legend(loc='upper left')
    plt.grid()
    plt.pause(0.001)


def plot_steps(steps_per_episode,  window_size=10):
    plt.figure(2)
    plt.clf()
    episode_number = len(steps_per_episode)
    smoothed_accuracies = [moving_average(steps_per_episode[:i+1], window_size) for i in range(episode_number)]

    plt.title('Number of Steps over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.plot(range(episode_number), steps_per_episode, label='Steps per Episode', color='blue')
    plt.plot(range(episode_number), smoothed_accuracies, label=f'Moving Avg ({window_size} episodes)', color='red')
    plt.hlines(20,0, episode_number, colors='red', linestyles=':', label='Perfect Steps')
    plt.legend(loc='upper left')
    plt.grid()
    plt.pause(0.001)


def calculate_optimal_steps(start, goal):
    """Calculate Manhattan distance between start and goal."""
    return abs(goal[0] - start[0]) + abs(goal[1] - start[1])

rewards = 0
#num_actions = 0
punishment = 0
optimal_path_count = 0
steps_per_episode = []
difference = []
# MAIN LOOP
for i_episode in range(N_EPISODES):
    env.reset()
    init_screen = get_screen()  # Get the initial screen state
    screens = deque([init_screen] * FRAMES, FRAMES)  # Initialize the deque with frames
    '''if N_EPISODES % 70 == 0:
        model.stdp1.normalize_weights()
        model.anti_stdp1.normalize_weights()
        model.stdp2.normalize_weights()
        model.anti_stdp2.normalize_weights()
        model.stdp3.normalize_weights()
        model.anti_stdp3.normalize_weights()'''

    # Calculate the optimal steps
    optimal_steps = calculate_optimal_steps(env.current_position, env.goal_position)
    steps_no = 0
    num_actions = 0
    total_reward = 0
    
    for t in count():
        state = torch.cat(list(screens), dim=1)  # Stack the frames for SNN input
        action = model.forward(state)  # Get the action from the SNN
        num_actions += 1

        print('Action: ', action)
        steps_no += 1
        #print('Agent Steps: ', steps_no)
        
        # Take action in the environment
        next_state, current_reward, done, _ = env.step(action)
        
        # Append the screen state to the frames
        screens.append(get_screen())  # Capture and append the new state
        next_state = torch.cat(list(screens), dim=1) if not done else None

        if current_reward > 0:
            model.reward(current_reward)
            total_reward += current_reward
            rewards += 1
        else:
            model.punish(current_reward)
            punishment +=1
            total_reward += current_reward

        
        if done:
            if steps_no == 20:
                optimal_path_count += 1
                print('Optimal Path Count: ', optimal_path_count)
                print('total number of agent steps: ' ,steps_no)
            else:
                total_steps = steps_no
                print('total number of agent steps: ',total_steps)
            
            difference.append(steps_no - optimal_steps)

            steps_per_episode.append(steps_no) 
            # Calculate and log accuracy
            accuracy = total_reward / (num_actions + 1e-10)
            print('Accuracy: ', accuracy)
            episode_accuracies.append(total_reward)

            plot_accuracy(episode_accuracies, window_size=10)
            #plot_steps(steps_per_episode)
            #plot_difference(difference)
            break


#print('Total Number of rewards: ', rewards)
#print('Total Number of punishments: ', punishment)
print('Optimal Path Count: ', optimal_path_count)
print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()