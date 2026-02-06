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
            
            if self.W_res_args["weights_distribution"] == "tile":
                self.G_res = self.tiled_rectangular(m, n)
            else:
                self.G_res = self.rectangular(m, n, neighborhood=self.W_res_args["neighborhood"])
                self.G_res = self._make_weights_negative(self.G_res, self.W_res_args["sign_frac"])
                self.G_res = self._make_graph_directed(self.G_res, self.W_res_args["directed_edges_fraction"], self.W_res_args["directed_edges_weights"])
            self._self_connection(self.G_res)
            W_res = nx.to_numpy_array(self.G_res)
            return torch.FloatTensor(W_res)
        else:
            #W_res = (torch.randint(0, 2, (self.size, self.size), dtype=torch.float32) * 2) - 1
            W_res = torch.rand(self.size, self.size) * 2 - 1
            W_res[torch.rand(self.size, self.size) > 0.1] = 0.0
            W_res[torch.eye(self.size) == 1] = self.W_res_args["self_connection"]
            return W_res

    def tiled_rectangular(self, m, n):
        tile_conf = self.W_res_args["tile"]
        tile_rows, tile_cols = tile_conf["shape"]
        neighborhood = self.W_res_args.get("neighborhood", "Von_Neumann")

        # Build periodic tile so all possible neighbor offsets have defined weights
        tile_G = self.tetragonal([tile_rows, tile_cols], periodic=True, neighborhood=neighborhood)
        if tile_conf["method"] == "random":
            for u, v, d in tile_G.edges(data=True):
                d['weight'] = np.random.random()

        # Apply sign_frac and directed edges to the tile so both patterns repeat
        self._make_weights_negative(tile_G, self.W_res_args["sign_frac"])
        tile_G = self._make_graph_directed(tile_G, self.W_res_args["directed_edges_fraction"], self.W_res_args["directed_edges_weights"])

        # Build directed tile weight lookup
        tile_weights = {}
        for u, v, d in tile_G.edges(data=True):
            tile_weights[(u, v)] = d['weight']

        # Build full lattice and convert to directed
        G = self.rectangular(m, n, neighborhood=neighborhood).to_directed()

        # Map full graph edges onto the tile via modulo (directed)
        for u, v, d in G.edges(data=True):
            u_r, u_c = u
            v_r, v_c = v
            t_u = (u_r % tile_rows, u_c % tile_cols)
            t_v = (v_r % tile_rows, v_c % tile_cols)
            if (t_u, t_v) in tile_weights:
                d['weight'] = tile_weights[(t_u, t_v)]
            else:
                d['weight'] = 0  # Fallback; should not happen with periodic tile

        return G


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
            if self.W_res_args["weights_distribution"] == "constant":
                d['weight'] = 1
            elif self.W_res_args["weights_distribution"] == "random":
                d['weight'] = np.random.random()
            elif self.W_res_args["weights_distribution"] == "custom":
                d['weight'] = 1 
            else:
                d['weight'] = 1

        return G

    def _self_connection(self, G):
        weight = self.W_res_args["self_connection"]
        for n in G:
            G.add_edge(n, n, weight=weight)
        return G
    
    def _make_graph_directed(self, G, dir_frac, dir_weights):
        bidir_edges = G.edges()
        dir_G =  G.to_directed()

        for u,v in bidir_edges:
            if np.random.random() < dir_frac:
                del_u, del_v = (u,v) if np.random.random() < 0.5 else (v,u)
                dir_G.edges[del_u, del_v]['weight'] *= dir_weights
            

        return dir_G
    
    def _make_weights_negative(self, G, sign_frac):
        for u, v, d in G.edges(data=True):
            sign = -1 if np.random.random() < sign_frac else 1
            d['weight'] = d['weight']*sign if 'weight' in d else sign
        return G

if __name__ == "__main__":
    matrix = Matrix({
        "size": 36,
        "W_in_args": {"input_scale": 1, "distribution": "uniform"},
        "W_res_args": {
            "self_connection": 0.0,
            "sign_frac": 0.5,
            "lattice": True,
            "neighborhood": "Moore",
            "weights_distribution": "tile",
            "directed_edges_weights": 0.0,
            "directed_edges_fraction": 0.5,
            "tile": {"shape": [3, 3], "method": "random"},
        },
    })

    G = matrix.G_res
    # Use grid positions so the lattice structure is visible
    pos = {node: (node[1], -node[0]) for node in G.nodes()}

    # Assign a color to each unique weight (rounded to avoid float noise)
    edge_weights = [round(d['weight'], 6) for _, _, d in G.edges(data=True)]
    unique_weights = sorted(set(edge_weights))
    cmap = plt.cm.get_cmap("tab20", len(unique_weights))
    weight_to_color = {w: cmap(i) for i, w in enumerate(unique_weights)}
    edge_colors = [weight_to_color[w] for w in edge_weights]

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=300)
    nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax, edge_color=edge_colors, width=2,
        connectionstyle="arc3,rad=0.15", arrows=True, arrowsize=15,
    )

    # Legend mapping colors to weights
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=weight_to_color[w], lw=2, label=f"{w:.3f}")
        for w in unique_weights
    ]
    ax.legend(handles=legend_handles, title="Edge weight", loc="upper left", fontsize=7)

    plt.title("Tiled lattice reservoir")
    plt.tight_layout()
    plt.show()
    
