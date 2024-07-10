import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network for the policy
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# Initialize the networks for π_theta and π_ref
input_dim = 4  # Example input dimension (e.g., state space in RL)
hidden_dim = 128
output_dim = 2  # Example output dimension (e.g., action space in RL)

# Initialize the policy networks
pi_theta = PolicyNetwork(input_dim, hidden_dim, output_dim)
pi_ref = PolicyNetwork(input_dim, hidden_dim, output_dim)

# Load the pre-trained weights into both networks
pretrained_path = 'pi_ref.pth'
pi_ref.load_state_dict(torch.load(pretrained_path))
pi_theta.load_state_dict(torch.load(pretrained_path))

# Set up the optimizer
optimizer = optim.Adam(pi_theta.parameters(), lr=1e-3)

# Dummy data for demonstration purposes
# Suppose we have states and actions from the environment
states = torch.randn(10, input_dim)
actions = torch.randint(0, output_dim, (10,))

# Loss function implementation based on DPO
def dpo_loss(pi_theta, pi_ref, states, actions):
    # Forward pass to get action probabilities
    pi_theta_probs = pi_theta(states)
    pi_ref_probs = pi_ref(states)

    # Select the probabilities of the taken actions
    pi_theta_selected_probs = pi_theta_probs.gather(1, actions.unsqueeze(1)).squeeze()
    pi_ref_selected_probs = pi_ref_probs.gather(1, actions.unsqueeze(1)).squeeze()

    # Compute the DPO loss
    log_ratios = torch.log(pi_theta_selected_probs / pi_ref_selected_probs)
    loss = -torch.mean(log_ratios)
    
    return loss

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = dpo_loss(pi_theta, pi_ref, states, actions)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print("Training completed.")
