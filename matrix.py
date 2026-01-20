import torch
import networkx as nx
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


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
        match self.W_in_args["distribution"]:
            case "uniform":
                W_in = torch.rand(self.size) - 0.5
            case "fixed":
                W_in = torch.ones(self.size)
            case _:
                raise ValueError("Invalid distribution")

        W_in *= self.W_in_args["input_scale"]
        return W_in

    def _init_W_res(self):
        if self.W_res_args["lattice"]:
            m = int(sqrt(self.size))
            n = int(sqrt(self.size))
            self.G_res = self.rectangular(m, n, neighborhood=self.W_res_args["neighborhood"])
            self.G_res = self._make_weights_negative(self.G_res, self.W_res_args["sign_frac"])

            self.G_res = self._make_graph_directed(self.G_res, self.W_res_args["directed_edges_fraction"])
            self._self_connection(self.G_res)
            W_res = nx.to_numpy_array(self.G_res)
            return torch.FloatTensor(W_res)
        else:
            #W_res = (torch.randint(0, 2, (self.size, self.size), dtype=torch.float32) * 2) - 1
            W_res = torch.rand(self.size, self.size) * 2 - 1
            W_res[torch.rand(self.size, self.size) > 0.1] = 0.0
            W_res[torch.eye(self.size) == 1] = self.W_res_args["self_connection"]
            return W_res

    def tetragonal(self, dim, periodic=False, neighborhood="Von_Neumann", dist_function=None):
        G = nx.grid_graph(dim, periodic=periodic)

        if neighborhood == "Moore":
            m, n = dim
            for i in range(m):
                for j in range(n):
                    u = (i, j)
                    # Diagonals: (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
                    diags = [(i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
                    for di, dj in diags:
                        if periodic:
                            v = (di % m, dj % n)
                            G.add_edge(u, v)
                        elif 0 <= di < m and 0 <= dj < n:
                            v = (di, dj)
                            G.add_edge(u, v)

        pos = dict(zip(G, G))
        nx.set_node_attributes(G, pos, 'pos')

        return G    

    def rectangular(self, m, n, periodic=False, neighborhood="Von_Neumann", dist_function=None):
        G = self.tetragonal([m, n], periodic=periodic, neighborhood=neighborhood)

        for n in G:
            pos = G.nodes[n]['pos']
            G.nodes[n]['pos'] = (pos[0], pos[1])

        for u, v, d in G.edges(data=True):
            d['weight'] = 1#/euclidean(G.nodes[u]['pos'], G.nodes[v]['pos'])

        return G

    def _self_connection(self, G):
        weight = self.W_res_args["self_connection"]
        for n in G:
            G.add_edge(n, n, weight=weight)
        return G
    
    def _make_graph_directed(self, G, dir_frac):
        bidir_edges = G.edges()
        dir_G =  G.to_directed()

        for u,v in bidir_edges:
            if np.random.random() < dir_frac:
                del_u, del_v = (u,v) if np.random.random() < 0.5 else (v,u)
                dir_G.remove_edge(del_u, del_v)

        return dir_G
    
    def _make_weights_negative(self, G, sign_frac):
        for u, v, d in G.edges(data=True):
            sign = -1 if np.random.random() < sign_frac else 1
            d['weight'] = d['weight']*sign if 'weight' in d else sign
        return G

if __name__ == "__main__":
    matrix = Matrix({"size": 16, "W_in_args": {"input_scale": 1, "distribution": "uniform"}, "W_res_args": {"self_connection": 1.0, "directed": 0.0, "sign_frac": 0.0, "lattice": True, "neighborhood": "Moore"}})
    pos = nx.spring_layout(matrix.G_res)
    nx.draw(matrix.G_res, pos=pos, with_labels=True, arrows=True)
    edge_labels = nx.get_edge_attributes(matrix.G_res, "weight")
    nx.draw_networkx_edge_labels(matrix.G_res, pos=pos, edge_labels=edge_labels)
    plt.show()
    print(matrix.W_res)
