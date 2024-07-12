
---

# GCN Karate Club Visualization

This project demonstrates the implementation and training of a Graph Convolutional Network (GCN) on the Karate Club dataset, visualizing the embeddings and graph structure during the training process. The animations of the graph and 3D embeddings are saved as GIFs.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GCN-Karate-Club-Visualization.git
   cd GCN-Karate-Club-Visualization
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:
   ```bash
   python main.py
   ```

2. The script will:
   - Load the Karate Club dataset and assign random countries (labels) to the nodes.
   - Initialize and train a GCN model.
   - Generate and save two GIF animations:
     - `graph_animation.gif`: Visualization of the graph structure and node labels during training.
     - `embedding_animation.gif`: Visualization of the 3D embeddings during training.

## Files

- `main.py`: Main script that runs the entire process: loading data, training the model, and generating visualizations.
- `methods_and_models.py`: Contains functions and classes for data loading, model definition, training, and visualization.
- `requirements.txt`: Lists all the dependencies required for the project.

## Requirements

The project requires the following packages:

- torch
- torch_geometric
- matplotlib
- networkx
- numpy

All required packages are listed in `requirements.txt`.

## Acknowledgements

- This project uses the Karate Club dataset provided by PyTorch Geometric.
- The implementation of the GCN model is based on PyTorch and PyTorch Geometric.

---