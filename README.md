## Project Name: Real-Time Traffic Flow Prediction Using Spatio-Temporal Graph Neural Networks

- Developed a Real-Time Traffic Prediction System:
  - Utilized Graph Neural Networks (GNN) and Graph Convolutional Networks (GCN) to model and predict traffic flow on road networks.
  - Focused on leveraging the spatial and temporal characteristics of traffic data for accurate predictions.

- Implemented a Spatio-Temporal GCN Model:
  - Combined GCN layers to capture spatial dependencies between road intersections.
  - Incorporated LSTM layers to model temporal dependencies and predict future traffic conditions.
  - Trained using PyTorch Geometric to efficiently handle graph-structured data.

- Simulated and Processed Traffic Data:
  - Generated synthetic traffic data on a 5x5 grid network representing road intersections.
  - Converted the NetworkX graph and traffic data into a PyTorch Geometric format for model training.
  - Achieved real-time prediction by continuously updating the model with new traffic data and visualizing predictions.

## Technologies and Tools Used:

- Python: Programming language for implementation and data manipulation.
- PyTorch Geometric: Library for implementing graph neural networks and handling graph data.
- NetworkX: Library for creating and manipulating complex networks and graphs.
- NumPy: Used for numerical computations and handling traffic data arrays.
- Matplotlib: Visualized traffic data and predictions for real-time monitoring and analysis.
