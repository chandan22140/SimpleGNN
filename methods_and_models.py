import torch


from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import KarateClub
import numpy as np
from torch_geometric.data import Data


from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation



def load_karate_club():
    dataset = KarateClub()
    data = dataset[0]
    
    # Number of countries
    num_countries = 4

    # Assign students to countries (labels)
    np.random.seed(42)
    countries = torch.tensor(np.random.choice(num_countries, data.num_nodes))

    # Update the labels in the data object
    data.y = countries

    return data



class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gcn = GCNConv(input_dim, 3)
        self.out = Linear(3, output_dim)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        z = self.out(h)
        return h, z




def train_model(model, data, epochs=200, lr=0.02):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Data for animations
    embeddings = []
    losses = []
    accuracies = []
    outputs = []

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        h, z = model(data.x, data.edge_index)
        loss = criterion(z, data.y)
        acc = accuracy(z.argmax(dim=1), data.y)
        loss.backward()
        optimizer.step()
        embeddings.append(h)
        losses.append(loss)
        accuracies.append(acc)
        outputs.append(z.argmax(dim=1))
        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')

    return embeddings, losses, accuracies, outputs

def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)


def visualize_graph(data, embeddings, losses, accuracies, outputs):
    def animate(i):
        G = to_networkx(data, to_undirected=True)
        nx.draw_networkx(G,
                         pos=nx.spring_layout(G, seed=0),
                         with_labels=True,
                         node_size=800,
                         node_color=outputs[i].numpy(),
                         cmap="hsv",
                         vmin=-2,
                         vmax=3,
                         width=0.8,
                         edge_color="grey",
                         font_size=14
                         )
        plt.title(f'Epoch {i} | Loss: {losses[i].item():.2f} | Acc: {accuracies[i].item()*100:.2f}%',
                  fontsize=18, pad=20)

    fig = plt.figure(figsize=(12, 12))
    plt.axis('off')
    anim = animation.FuncAnimation(fig, animate, np.arange(0, 200, 10), interval=500, repeat=True)
    
    # Save the animation using Pillow writer
    try:
        anim.save('graph_animation.gif', writer='pillow', fps=2)
    except Exception as e:
        print(f"An error occurred with saving the animation: {e}")

def visualize_3d(embeddings, losses, accuracies, outputs, data):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def animate_3d(i):
        embed = embeddings[i].detach().cpu().numpy()
        ax.clear()
        ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
                   s=200, c=data.y, cmap="hsv", vmin=-2, vmax=3)
        plt.title(f'Epoch {i} | Loss: {losses[i].item():.2f} | Acc: {accuracies[i].item()*100:.2f}%',
                  fontsize=18, pad=40)

    anim = animation.FuncAnimation(fig, animate_3d, np.arange(0, 200, 10), interval=800, repeat=True)
    
    # Save the animation using Pillow writer
    try:
        anim.save('embedding_animation.gif', writer='pillow', fps=2)
    except Exception as e:
        print(f"An error occurred with saving the animation: {e}")


