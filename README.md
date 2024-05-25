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


<img width="528" alt="Screenshot 2024-05-25 at 10 03 59 AM" src="https://github.com/Arnavsao/Real-Time-Traffic-Flow-Prediction-Using-Spatio-Temporal-GNN/assets/140349606/6f2fb682-90c0-4f97-afa2-2652cacd560f">



## Technologies and Tools Used:

- Python: Programming language for implementation and data manipulation.
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/200px-Python-logo-notext.svg.png" alt="Python" width="100"/>

- PyTorch Geometric: Library for implementing graph neural networks and handling graph data.
  <img src="https://pytorch-geometric.readthedocs.io/en/latest/_images/pyg_logo_text.svg" alt="PyTorch Geometric" width="200"/>

- NetworkX: Library for creating and manipulating complex networks and graphs.
  <img src="https://networkx.org/_static/networkx_logo.svg" alt="NetworkX" width="200"/>

- NumPy: Used for numerical computations and handling traffic data arrays.
  <img src="https://numpy.org/images/logos/numpy.svg" alt="NumPy" width="100"/>

- Matplotlib: Visualized traffic data and predictions for real-time monitoring and analysis.
  <img src="https://matplotlib.org/stable/_static/logo2_compressed.svg" alt="Matplotlib" width="200"/>
