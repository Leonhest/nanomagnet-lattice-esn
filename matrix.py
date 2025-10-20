import torch
import networkx as nx
from math import sqrt
import matplotlib.pyplot as plt


def euclidean(x, y):
    """
    The euclidean distance metric that is used within NetworkX.
    """
    return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

class Matrix:
    def __init__(self, conf):
        self.size = conf["size"]
        self.W_in_args = conf["W_in_args"]
        self.W_res_args = conf["W_res_args"]
        self.W_in = self._init_W_in()
        self.W_res = self._init_W_res()

    def _init_W_in(self):
        W_in = torch.ones(self.size)
        W_in *= self.W_in_args["input_scale"]
        return W_in

    def _init_W_res(self):
        m = int(sqrt(self.size))
        n = int(sqrt(self.size))
        self.G_res = self.rectangular(m, n)
        W_res = nx.to_numpy_array(self.G_res)
        return torch.FloatTensor(W_res)

    def tetragonal(self, dim, periodic=False, dist_function=None):
        G = nx.grid_graph(dim, periodic=periodic)

        pos = dict(zip(G, G))
        nx.set_node_attributes(G, pos, 'pos')

        return G    

    def rectangular(self, m, n, rect_ratio=1.0, periodic=False, dist_function=None):
        G = self.tetragonal([m, n], periodic=periodic)

        for n in G:
            pos = G.nodes[n]['pos']
            G.nodes[n]['pos'] = (pos[0], pos[1]*rect_ratio)

        for u, v, d in G.edges(data=True):
            d['weight'] = 1/euclidean(G.nodes[u]['pos'], G.nodes[v]['pos'])

        return G

if __name__ == "__main__":
    matrix = Matrix({"size": 81, "W_in_args": {"input_scale": 1}, "W_res_args": {}})
    nx.draw(matrix.G_res, pos=nx.spring_layout(matrix.G_res), with_labels=True)
    plt.show()