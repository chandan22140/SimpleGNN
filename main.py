
from methods_and_models import load_karate_club,  GCN, train_model, visualize_graph, visualize_3d

# Load data
data = load_karate_club()

# Initialize model
model = GCN(input_dim=data.num_features, output_dim=4)

# Train model
embeddings, losses, accuracies, outputs = train_model(model, data)

# Visualize results
visualize_graph(data, embeddings, losses, accuracies, outputs)
visualize_3d(embeddings, losses, accuracies, outputs, data)
