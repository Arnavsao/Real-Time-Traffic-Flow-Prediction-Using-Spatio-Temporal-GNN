# Road Traffic Prediction using SpatioTemporalGCN

This project involves simulating and predicting road traffic flow using a SpatioTemporal Graph Convolutional Network (GCN). The goal is to predict future traffic flow dynamics based on historical data from a road network graph.

## Overview

The project consists of the following steps:

1. **Creating Road Network Graph**: Generates a simple road network graph using NetworkX with a 5x5 grid structure.

2. **Generating Traffic Data**: Simulates random traffic flow data for the nodes in the graph.

3. **Visualizing the Graph**: Visualizes the road network graph using Matplotlib.

4. **Converting to PyTorch Geometric Data**: Converts the NetworkX graph to a PyTorch Geometric data object and assigns traffic data as node features.

5. **Defining SpatioTemporalGCN Model**: Defines a SpatioTemporalGCN model using PyTorch Geometric, which consists of Graph Convolutional layers followed by an LSTM layer and a linear layer for prediction.

6. **Preparing Data for Training**: Prepares the dataset for training, creating a DataLoader for batches of graphs.

7. **Training Loop**: Trains the SpatioTemporalGCN model using MSE loss and Adam optimizer.

8. **Real-Time Prediction Simulation**: Simulates real-time prediction of future traffic flow by iteratively predicting traffic for subsequent time steps and updating the input data.

## Usage

To run the project, follow these steps:

1. Install the required dependencies: `networkx`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `torch`.

2. Execute the Python script provided in this repository.

3. The script will generate traffic data, train the SpatioTemporalGCN model, and simulate real-time predictions.

## Code

```python
# Add the Python code here
