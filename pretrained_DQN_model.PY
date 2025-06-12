import gym
import torch
import os
from collections import deque
import matplotlib.pyplot as plt
from cartpole_vision_v1 import *

# Load saved model parameters
LOAD_MODEL = True  # Set to True for loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Initialize the Gym environment
env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped

# Get screen dimensions and number of actions
n_actions = env.action_space.n
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Initialize the policy and target networks
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)

# Load the trained model weights
if LOAD_MODEL:
    policy_net_checkpoint = torch.load('save_model/policy_net_final.pt')
    target_net_checkpoint = torch.load('save_model/target_net_final.pt')
    policy_net.load_state_dict(policy_net_checkpoint)
    target_net.load_state_dict(target_net_checkpoint)
    policy_net.eval()  # Set the network to evaluation mode
    target_net.eval()

# Function to select an action based on the state
def select_action(state, stop_training):
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)  # Greedy action selection

# Collect new data
num_episodes = 800  # Number of episodes to run with the loaded model
episode_durations = []
new_image_directory = "saved_trained_images"  
os.makedirs(new_image_directory, exist_ok=True) 

for i_episode in range(num_episodes):
    env.reset()
    init_screen = get_screen()

    screens = deque([init_screen] * FRAMES, FRAMES)
    state = torch.cat(list(screens), dim=1)

    for t in range(1000):  # Run for a max of 1000 time steps
        action = select_action(state, stop_training=False)
        #print(action)

        # Take action in the environment
        state_variables, _, done, _, _ = env.step(action.item())

        # Save the state as an image
        if state is not None:  # Check if state is valid
            # Handle the case for a batch vs single image
            if state.dim() == 4:  # Batch of images
                state_np = state[0].permute(1, 2, 0).cpu().numpy()  # Get the first image
            elif state.dim() == 3:  # Single image
                state_np = state.permute(1, 2, 0).cpu().numpy()
            else:
                raise ValueError("Unexpected dimension for state tensor")

            state_np = (state_np * 255).astype(np.uint8)  # Convert to uint8

            # Create a PIL Image
            img = Image.fromarray(state_np)

            # Create a subdirectory for the action if it doesn't exist
            action_item = action.item()
            print(f"Saving image for action: {action_item}")  # Log action to confirm
            action_directory = os.path.join(new_image_directory, f'action_{action_item}')
            os.makedirs(action_directory, exist_ok=True)
            image_filename = os.path.join(action_directory, f"episode_{i_episode}_step_{t}.png")
            img.save(image_filename)
            print(f"Image saved as {image_filename}")
        # Observe new state
        screens.append(get_screen())  # Capture new screen
        next_state = torch.cat(list(screens), dim=1) if not done else None

        # Update state
        state = next_state

        # Break if the environment signals done
        if done:
            episode_durations.append(t + 1)
            print(f'Episode {i_episode + 1} finished after {t + 1} timesteps')
            break
# Plot the results
plt.plot(episode_durations)
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.title('Episode Duration with Loaded Model')
plt.show()

env.close()