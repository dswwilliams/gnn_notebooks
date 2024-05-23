import torch_geometric
import networkx as nx
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T


from torch_geometric.datasets import ShapeNet
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_graph_shapenet(data):
    x,y,z = data.pos[:,0], data.pos[:,1], data.pos[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='b', marker='o')

    fig.tight_layout()
    # square aspect ratio
    ax.set_aspect('equal')
    plt.show()

def get_graph():
    dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'], pre_transform=T.KNNGraph(k=6))
    data = dataset[0]  # Get the first graph object.

    return data


if __name__ == "__main__":
    graph = get_graph()
    plot_graph_shapenet(graph)