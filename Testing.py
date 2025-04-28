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
from functional import *
from utils import *
from torchvision import transforms
from PIL import Image
from gym import spaces
import random

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
N_EPISODES = 1000
LAST_EPISODES_NUM = 20
FRAMES = 2
RESIZE_PIXELS = 60
eps_decay_rate = 0.99

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

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

        self.goal_position = (11, 11)
        self.current_position = (0, 0)
        self.previous_position = (0, 0)

        self.grid_size = 12
        
        #self.blocks = self.generate_fixed_blocks()
        self.blocks = self.generate_random_blocks()
        self.trees = self.generate_tree_positions()

        self.fig, self.ax = plt.subplots()
        self.state = self.get_state()

        self.steps = 0  # Timer counter
        self.max_steps = 22  # Set a limit (adjust as needed)
        self.tree_visited = False


    
    '''def generate_fixed_blocks(self):
        fixed_blocks = {
        #(2, 3),
        (10, 5),
        #(4, 0),
        #(7, 7),
        (4,5),
        #(7,2),
        (8,3),
        (0,11)
        #(11,0),
        #(11,6)  
       }

        return fixed_blocks'''
    
    def generate_random_blocks(self, num_blocks=2, min_distance=2):
        """ Generate random block positions ensuring they are at least 'min_distance' apart. """
        blocks = set()
    
        while len(blocks) < num_blocks:
            block_position = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
        
        # Ensure block is not at the goal, start position, and is far enough from other blocks
            if block_position != self.goal_position and block_position != self.current_position:
                if all(np.linalg.norm(np.array(block_position) - np.array(b)) >= min_distance for b in blocks):
                    blocks.add(block_position)
                
        return blocks

   

    
    def generate_tree_positions(self):
    # Generate nearby positions within Manhattan distance <= 3
        nearby_positions = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if abs(dx) + abs(dy) <= 3 and (dx != 0 or dy != 0):  # exclude the goal itself
                    new_x = self.goal_position[0] + dx
                    new_y = self.goal_position[1] + dy
                    if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                        nearby_positions.append((new_x, new_y))

    # Sort by distance to goal (closest first)
        nearby_positions.sort(key=lambda pos: abs(pos[0] - self.goal_position[0]) + abs(pos[1] - self.goal_position[1]))

    # Return the first 3
        return nearby_positions[:0]

    def reset(self):
        # Reset the agent's position to start at (0, 0)
        self.current_position = (0, 0)
        self.previous_position = (0, 0)
        
        # Re-generate random blocks each time we reset
        
        #self.blocks = self.generate_fixed_blocks()
        #self.blocks = self.generate_random_blocks()

        # Update the state
        self.state = self.get_state()  # Reset the state to the initial grid
        return self.state
    
    def relocate(self):
        # Reset the agent's position to start at (0, 0)
        self.current_position = (0, 0)
        self.previous_position = (0, 0)
        
        # Re-generate random blocks each time we reset
        
        #self.blocks = self.generate_fixed_blocks()
        self.blocks = self.generate_random_blocks()

        # Update the state
        self.state = self.get_state()  # Reset the state to the initial grid
        return self.state

    
    
    def step(self, action):
        done = False
        self.previous_position = self.current_position
        self.steps += 1 
    
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
        if self.steps >= 100:
            done = True
            current_reward -= 3

     #tree reward
    # if self.current_position == self.trees:
       # if not self.tree_visited:
        #    reward += 2  # Give reward only on first visit
         #   self.tree_visited = True  # Mark it as visited


    # Check if the next position hits a block (wall)
        if tuple(next_position) in self.blocks:
            current_reward = -2
            self.current_position = self.previous_position  # Stay in the previous position
        else:
            self.current_position = tuple(next_position)

        # Distance-based reward
        current_distance = np.linalg.norm(np.array(self.current_position) - np.array(self.goal_position))
        previous_distance = np.linalg.norm(np.array(self.previous_position) - np.array(self.goal_position))
        if current_distance < previous_distance:
            current_reward = 1
        else:
            current_reward = -1

        # Goal reached
        if self.current_position == self.goal_position:
            if self.steps <= self.max_steps:
                current_reward = 10
            else:
                current_reward = 5
                done = True
       
        # Additional time penalty for exceeding max steps but not reaching the goal
           # if self.steps > self.max_steps:
            #    reward -= 3  # Time penalty, but episode continues

          #  done = False  # The episode doesn't end

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
        wall_width = 0.5  # Width of the walls
        for block in self.blocks:
            block_x, block_y = block
        # Draw walls using rectangles
            self.ax.add_patch(plt.Rectangle((block_y - wall_width / 2, block_x - wall_width / 2), 
                                          wall_width, wall_width, color='blue'))  # Wall in blue

    # Plot the robot (agent) as a rectangle
        robot_size = 0.4  # Size of the robot
        self.ax.add_patch(plt.Rectangle((self.current_position[1] - robot_size / 2, 
                                      self.current_position[0] - robot_size / 2), 
                                      robot_size, robot_size, color='red', label='Robot'))  # Robot in red
        

    # Plot the goal's position
        goal_x, goal_y = self.goal_position
        self.ax.plot(goal_y, goal_x, 'go', markersize=10, label='Goal')  # Goal in green

        #Plot the trees (as green rectangles)
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
       screen = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

       agent_x, agent_y = self.current_position
       screen[agent_x, agent_y] = [255, 0, 0]  

       goal_x, goal_y = self.goal_position
       screen[goal_x, goal_y] = [0, 255, 0]  

       for block in self.blocks:
        block_x, block_y = block
        screen[block_x, block_y] = [0, 0, 255]  

    # Encode agent velocity as an additional feature
       velocity_x = agent_x - self.previous_position[0]
       velocity_y = agent_y - self.previous_position[1]
    
       velocity_channel = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
       velocity_channel[agent_x, agent_y] = int((velocity_x**2 + velocity_y**2) ** 0.5 * 255 / np.sqrt(2))
    
    # Stack velocity channel into RGB image
       screen = np.dstack((screen, velocity_channel))
       return transform(Image.fromarray(screen))

    
graph_name = 'CustomNavigationEnv_Stop-' + str(TRAINING_STOP) + '_LastEpNum-' + str(LAST_EPISODES_NUM)

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
        self.pool = Pooling(kernel_size=2, stride=1, padding=0)
        self.conv = Convolution(input_channels, self.number_of_features, 3, 0.8, 0.05)
        self.stdp = STDP(conv_layer=self.conv, learning_rate=(0.0001, -0.0009), use_stabilizer=True, lower_bound=0, upper_bound=1)
        self.anti_stdp = STDP(conv_layer=self.conv, learning_rate=(-0.009, 0.00001), use_stabilizer=True, lower_bound=0, upper_bound=1)
        
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

        if len(winners) == 0:
            return np.random.randint(0, 4)  # Choose a random valid action
        return self.decision_map[winners[0][0]]


    def reward(self, current_reward):
        if current_reward > 0:
            self.stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
            print('Reward')


    def punish(self, current_reward):
        if current_reward <= 0:
            self.anti_stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
            print('Punishment')
   

def calculate_optimal_steps(start, goal):
    """Calculate Manhattan distance between start and goal."""
    return abs(goal[0] - start[0]) + abs(goal[1] - start[1])

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
learned_weights = None

steps_done = 0
episode_accuracies = []

def moving_average(data, window_size=10):
    """Compute moving average over the last 'window_size' elements."""
    if len(data) < window_size:
        return np.mean(data)  # Use mean of available values if not enough data
    return np.mean(data[-window_size:])  # Compute moving average of last N elements

# New plotting function for accuracy
def plot_accuracy(accuracy_scores, window_size=10):
    plt.figure(2)
    plt.clf()
    episode_number = len(accuracy_scores)
    smoothed_accuracies = [moving_average(accuracy_scores[:i+1], window_size) for i in range(episode_number)]
    
    plt.title('Average Reward per Action')
    plt.xlabel('Episode')
    plt.ylabel('Average Rewards')
    plt.plot(range(episode_number), accuracy_scores, label='Raw Accuracy', color='blue', alpha=0.4)
    plt.plot(range(episode_number), smoothed_accuracies, label=f'Moving Avg ({window_size} episodes)', color='red')
    plt.legend(loc='upper left')
    plt.grid()
    
    if is_ipython:
        display.clear_output(wait=True)  # Clear previous output
        display.display(plt.gcf())  # Display updated figure
    else:
        plt.pause(0.001)


def plot_difference(difference, window_size=10):
    plt.figure(2)
    plt.clf()
    episode_number = len(difference)
    smoothed_accuracies = [moving_average(difference[:i+1], window_size) for i in range(episode_number)]

    plt.title('Difference from optimal steps over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Difference')
    plt.plot(range(episode_number), difference, label='Differences from Optimal Steps', color='blue', alpha=0.4)
    plt.plot(range(episode_number), smoothed_accuracies, label=f'Moving Avg ({window_size} episodes)', color='red')
    plt.hlines(0, 0, episode_number, colors='red', linestyles=':', label='No difference')
    plt.legend(loc='upper left')
    plt.grid()
    if is_ipython:
        display.clear_output(wait=True)  # Clear previous output
        display.display(plt.gcf())  # Display updated figure
    else:
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


episode_rewards = []
milestone_info = []  # To store (block_config_index, episode_number)

block_config_cycles = 20
episodes_per_cycle = 1000
total_episodes = block_config_cycles * episodes_per_cycle

steps_per_episode = []
ptimal_path_count = 0
difference = []
optimal_path_count = 0
logged_configs = {}

# MAIN LOOP
for config_idx in range(20):
    env.relocate()

    for i_episode in range(1000):
        env.reset()
        init_screen = get_screen()  # Get the initial screen state
        screens = deque([init_screen] * FRAMES, FRAMES)  # Initialize the deque with frames
        tree_visited = False

        total_reward = 0
        num_actions = 0
        steps_no = 0
   # Load previously learned weights
        if learned_weights is not None:
            with torch.no_grad():
                if learned_weights.shape == model.conv.weight.shape:
                    model.conv.weight.copy_(learned_weights)
                else:
                    print("Error: Shape mismatch between learned weights and model weights")

        optimal_steps = calculate_optimal_steps(env.current_position, env.goal_position)



        for t in count():
            state = torch.cat(list(screens), dim=1)  # Stack the frames for SNN input
       # action = select_action(state)  # Get the action from the SNN
            action = model.forward(state) 
            print('Action: ', action)
            num_actions += 1
            steps_no += 1
       
        
        # Take action in the environment
            next_state, current_reward, done, _ = env.step(action)
        
        # Append the screen state to the frames
            screens.append(get_screen())  # Capture and append the new state
            next_state = torch.cat(list(screens), dim=1) if not done else None

            if current_reward > 0:
                model.reward(current_reward)
                total_reward += current_reward
            else:
                model.punish(current_reward)
                total_reward += current_reward

        
            if done:
                  # Store updated weights
                learned_weights = model.conv.weight.clone()
                
                if steps_no == optimal_steps:
                    optimal_path_count += 1
                    print('Optimal Path Count: ', optimal_path_count)
                    print('total number of agent steps: ' ,steps_no)
                else:
                    total_steps = steps_no
                    print('total number of agent steps: ',total_steps)

                difference.append(steps_no - optimal_steps)
            # Calculate and log accuracy
                Average_reward = total_reward / (num_actions + 1e-10)
                print('Average reward per action: ', Average_reward)
                if Average_reward >= 1.18:
                    with open("milestones.txt", "a") as f:
                        if config_idx not in logged_configs:
                            f.write(f"Config {config_idx} reached avg reward = 1.18 at episode {i_episode}\n")
                episode_accuracies.append(Average_reward)

                plot_accuracy(episode_accuracies, window_size=10)
                #steps_per_episode.append(steps_no) 
                #plot_steps(steps_per_episode, window_size = 10)
                 #plot_difference(difference, window_size = 10)
                
                print('optimal steps', optimal_steps)
                break
    
print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()