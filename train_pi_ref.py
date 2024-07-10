import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import ParameterGrid
import numpy as np

# Step 1: Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Continuous output, no activation here
        return x

# Step 2: Generate Dummy Data
np.random.seed(0)
torch.manual_seed(0)

input_dim = 4  # Example input dimension (e.g., state space in RL)
hidden_dim = 128
output_dim = 1  # Continuous output dimension

num_samples = 1000
X = np.random.randn(num_samples, input_dim).astype(np.float32)
y = np.random.randn(num_samples, output_dim).astype(np.float32)

# Create TensorDataset
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))

# Split into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Step 3: Hyperparameter Tuning and Training
def train_and_evaluate(train_loader, test_loader, hidden_dim, learning_rate, num_epochs):
    # Initialize the network
    model = PolicyNetwork(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()  # Mean Squared Error for continuous output
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
        test_loss /= len(test_loader)

    return test_loss, model

# Hyperparameter grid
param_grid = {
    'hidden_dim': [64, 128, 256],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'num_epochs': [50, 100]
}

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Perform hyperparameter tuning
best_params = None
best_test_loss = float('inf')
best_model = None

for params in ParameterGrid(param_grid):
    hidden_dim = params['hidden_dim']
    learning_rate = params['learning_rate']
    num_epochs = params['num_epochs']

    test_loss, model = train_and_evaluate(train_loader, test_loader, hidden_dim, learning_rate, num_epochs)
    
    print(f"Params: {params}, Test Loss: {test_loss}")

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_params = params
        best_model = model

print(f"Best Params: {best_params}, Best Test Loss: {best_test_loss}")

# Step 4: Save the Best Model
torch.save(best_model.state_dict(), 'pi_ref.pth')

# Step 5: Test Set Evaluation (optional)
best_model.eval()
with torch.no_grad():
    test_loss = 0
    for inputs, targets in test_loader:
        outputs = best_model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
    test_loss /= len(test_loader)

print(f"Final Test Loss: {test_loss}")
