import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate a simple road network graph
def create_road_network():
    G = nx.grid_2d_graph(5, 5)  # Create a 5x5 grid graph
    G = nx.convert_node_labels_to_integers(G)  # Convert node labels to integers
    return G

# Simulate traffic flow data
def generate_traffic_data(G, num_timesteps=100):
    num_nodes = G.number_of_nodes()
    traffic_data = np.random.rand(num_timesteps, num_nodes)  # Random traffic data
    return traffic_data

# Visualize the graph
def visualize_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    plt.show()

# Create and visualize the road network
G = create_road_network()
visualize_graph(G)

# Generate traffic data
traffic_data = generate_traffic_data(G)
print("Traffic data shape:", traffic_data.shape)  # Should be (num_timesteps, num_nodes)

import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Convert the NetworkX graph to a PyTorch Geometric data object
def convert_to_pyg_data(G, traffic_data):
    pyg_data = from_networkx(G)
    pyg_data.x = torch.tensor(traffic_data, dtype=torch.float).t()  # Traffic data as node features (transposed)
    return pyg_data

# Convert to PyTorch Geometric data format
pyg_data = convert_to_pyg_data(G, traffic_data)
print(pyg_data)

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SpatioTemporalGCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_timesteps):
        super(SpatioTemporalGCN, self).__init__()
        self.gcn1 = GCNConv(num_node_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_timesteps)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = x.unsqueeze(0)  # Add batch dimension
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x.squeeze(0))
        return x

# Define model parameters
num_node_features = pyg_data.num_node_features
hidden_dim = 32
num_timesteps = 10  # Predict next 10 timesteps

# Instantiate the model
model = SpatioTemporalGCN(num_node_features, hidden_dim, num_timesteps)
print(model)

import torch.optim as optim
from torch_geometric.data import DataLoader

# Prepare data for training (using the same data for simplicity)
# Normally, you'd split into training and test sets
dataset = [pyg_data for _ in range(100)]  # Create a dataset of 100 identical graphs

# Create data loader
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.x[-1, :])  # Compare to the last timestep
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Train the model
train(model, train_loader, criterion, optimizer)

import time

# Simulate real-time prediction
def real_time_prediction(model, initial_data, steps=10):
    model.eval()
    current_data = initial_data
    predictions = []

    for step in range(steps):
        with torch.no_grad():
            prediction = model(current_data)
            predictions.append(prediction)
            # Simulate new data coming in (use the last prediction as new data)
            new_data = torch.cat((current_data.x[:, 1:], prediction.unsqueeze(1)), dim=1)
            current_data.x = new_data

        # Simulate real-time delay
        time.sleep(1)
        print(f"Step {step+1}/{steps}, Prediction: {prediction.squeeze().tolist()}")

    return predictions

# Simulate initial data
initial_data = pyg_data.clone()
initial_data.x = torch.tensor(traffic_data[:num_timesteps, :], dtype=torch.float).t()

# Perform real-time prediction
predictions = real_time_prediction(model, initial_data)
