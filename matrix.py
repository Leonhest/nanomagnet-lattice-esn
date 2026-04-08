import json
import torch
import networkx as nx
from math import sqrt
import numpy as np


class OffsetTile:
    """Tile with per-(position, offset) weights.

    Avoids aliasing that occurs with nx.DiGraph on small periodic tiles
    where multiple offsets map to the same (src, dst) node pair.
    """

    def __init__(self, tile_shape, offset_weights, neighborhood=None):
        self.tile_shape = tile_shape          # (rows, cols)
        self.offset_weights = offset_weights  # {((r,c), (dr,dc)): float}
        self.neighborhood = neighborhood

    def number_of_edges(self):
        return len(self.offset_weights)

    def nodes(self):
        return [(r, c) for r in range(self.tile_shape[0])
                       for c in range(self.tile_shape[1])]

    def edges(self, data=False):
        if data:
            return [
                (pos, (pos[0] + off[0], pos[1] + off[1]), {"weight": w})
                for (pos, off), w in self.offset_weights.items()
            ]
        return [
            (pos, (pos[0] + off[0], pos[1] + off[1]))
            for pos, off in self.offset_weights.keys()
        ]


def euclidean(x, y):
    """
    The euclidean distance metric that is used within NetworkX.
    """
    return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))



def alternating_projections(W, iterations):
    """Push W toward orthogonal while preserving its zero pattern."""
    mask = (W != 0)
    for _ in range(iterations):
        U, _, Vh = np.linalg.svd(W, full_matrices=True)
        W = (U @ Vh).astype(np.float32)  # orthogonal projection
        W = W * mask                      # sparsity projection
    return W


def orthogonalize_tile(tile_G, method="exact", iterations=50):
    """Orthogonalize a tile graph's adjacency matrix and return a new tile graph."""
    nodes = sorted(tile_G.nodes())
    n = len(nodes)
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}

    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.float32)
    for u, v, d in tile_G.edges(data=True):
        A[node_to_idx[u], node_to_idx[v]] = d['weight']

    # Orthogonalize
    if method == "exact":
        U, _, Vh = np.linalg.svd(A, full_matrices=True)
        A = (U @ Vh).astype(np.float32)
    elif method == "alternating":
        A = alternating_projections(A, iterations)

    # Build new tile graph
    new_tile = nx.DiGraph()
    for nd in nodes:
        new_tile.add_node(nd)
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if abs(A[i, j]) > 1e-15:
                new_tile.add_edge(u, v, weight=float(A[i, j]))
    return new_tile


def save_tile(tile_G, path, metadata=None):
    """Save a tile graph or OffsetTile as JSON."""
    if isinstance(tile_G, OffsetTile):
        nodes = tile_G.nodes()
        data = {
            "format": "offset",
            "nodes": [list(n) for n in nodes],
            "edges": [
                {"src": list(pos), "offset": list(offset), "weight": weight}
                for (pos, offset), weight in tile_G.offset_weights.items()
            ],
        }
        meta = {"tile_shape": list(tile_G.tile_shape)}
        if tile_G.neighborhood is not None:
            meta["neighborhood"] = tile_G.neighborhood
    else:
        nodes = list(tile_G.nodes())
        tile_shape = [max(n[0] for n in nodes) + 1, max(n[1] for n in nodes) + 1]
        data = {
            "nodes": nodes,
            "edges": [
                {"src": _node_to_json(u), "dst": _node_to_json(v), "weight": d["weight"]}
                for u, v, d in tile_G.edges(data=True)
            ],
        }
        meta = {"tile_shape": tile_shape}
    if metadata:
        meta.update(metadata)
    data["metadata"] = meta
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_tile(path):
    """Load a tile graph from JSON."""
    tile_G, _ = load_tile_with_metadata(path)
    return tile_G


def load_tile_with_metadata(path):
    """Load a tile graph (or OffsetTile) and its metadata from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    meta = data.get("metadata", {})

    if data.get("format") == "offset":
        tile_shape = tuple(meta["tile_shape"])
        offset_weights = {}
        for edge in data["edges"]:
            pos = tuple(edge["src"])
            offset = tuple(edge["offset"])
            offset_weights[(pos, offset)] = edge["weight"]
        tile = OffsetTile(tile_shape, offset_weights,
                          neighborhood=meta.get("neighborhood"))
        return tile, meta

    tile_G = nx.DiGraph()
    for node in data["nodes"]:
        tile_G.add_node(_node_from_json(node))
    for edge in data["edges"]:
        tile_G.add_edge(
            _node_from_json(edge["src"]),
            _node_from_json(edge["dst"]),
            weight=edge["weight"],
        )
    return tile_G, meta


def _resolve_neighborhood_offsets(neighborhood):
    """Resolve a neighborhood specifier to a list of (di, dj) offsets."""
    if neighborhood == "Von_Neumann":
        return Matrix._shell_offsets(1)
    elif neighborhood == "Moore":
        return Matrix._shell_offsets(2)
    elif isinstance(neighborhood, int) and neighborhood >= 1:
        return Matrix._shell_offsets(neighborhood)
    else:
        raise ValueError(f"Unknown neighborhood: {neighborhood}")


def _tile_edges_to_lattice(tile_G, tile_rows, tile_cols, m, n, neighborhood=2):
    """Replicate tile weights across a lattice grid using modulo position mapping.

    For each lattice edge (determined by neighborhood offsets), the weight is
    looked up from the tile by mapping source and target positions via modulo.
    This correctly handles tiles smaller than the neighborhood radius.
    """
    tile_weights = {}
    for u, v, d in tile_G.edges(data=True):
        tile_weights[(u, v)] = d.get('weight', 1.0)

    offsets = _resolve_neighborhood_offsets(neighborhood)

    G = nx.DiGraph()
    for r in range(m):
        for c in range(n):
            G.add_node((r, c))

    for r in range(m):
        for c in range(n):
            t_u = (r % tile_rows, c % tile_cols)
            for dr, dc in offsets:
                tr, tc = r + dr, c + dc
                if 0 <= tr < m and 0 <= tc < n:
                    t_v = (tr % tile_rows, tc % tile_cols)
                    if (t_u, t_v) in tile_weights:
                        G.add_edge((r, c), (tr, tc), weight=tile_weights[(t_u, t_v)])

    return G


def _tile_edges_to_lattice_by_offset(tile_G, tile_rows, tile_cols, m, n):
    """Replicate tile edges by offset. Used for dense/orthogonalized tiles
    where edges may not correspond to standard neighborhood offsets."""
    G = nx.DiGraph()
    for r in range(m):
        for c in range(n):
            G.add_node((r, c))

    for u, v, d in tile_G.edges(data=True):
        dr = v[0] - u[0]
        dc = v[1] - u[1]
        if abs(dr) > tile_rows // 2:
            dr = dr - tile_rows if dr > 0 else dr + tile_rows
        if abs(dc) > tile_cols // 2:
            dc = dc - tile_cols if dc > 0 else dc + tile_cols
        for r in range(m):
            for c in range(n):
                if (r % tile_rows, c % tile_cols) == (u[0] % tile_rows, u[1] % tile_cols):
                    tr = r + dr
                    tc = c + dc
                    if 0 <= tr < m and 0 <= tc < n:
                        G.add_edge((r, c), (tr, tc), weight=d["weight"])

    return G


def _tile_offsets_to_lattice(offset_weights, tile_rows, tile_cols, m, n):
    """Tile an OffsetTile onto a full lattice using per-(position, offset) weights."""
    # Group by tile position for efficient lookup
    pos_offsets = {}
    for (pos, offset), weight in offset_weights.items():
        pos_offsets.setdefault(pos, []).append((offset, weight))

    G = nx.DiGraph()
    # Add all nodes first in row-major order to ensure consistent ordering
    for r in range(m):
        for c in range(n):
            G.add_node((r, c))
    # Then add edges
    for r in range(m):
        for c in range(n):
            t_pos = (r % tile_rows, c % tile_cols)
            for (dr, dc), weight in pos_offsets.get(t_pos, []):
                tr, tc = r + dr, c + dc
                if 0 <= tr < m and 0 <= tc < n:
                    G.add_edge((r, c), (tr, tc), weight=weight)
    return G


def tile_to_lattice(tile_G, tile_shape, lattice_size=36, neighborhood=2):
    """Tile a directed tile graph or OffsetTile onto a full lattice.

    Standalone version of Matrix._tile_from_graph for use outside the class.
    """
    tile_rows, tile_cols = tile_shape
    m = int(sqrt(lattice_size))
    n = int(sqrt(lattice_size))
    if isinstance(tile_G, OffsetTile):
        return _tile_offsets_to_lattice(tile_G.offset_weights, tile_rows, tile_cols, m, n)
    return _tile_edges_to_lattice(tile_G, tile_rows, tile_cols, m, n, neighborhood)


def _is_tile_path(value):
    """Check if a type value is a file path to a saved tile."""
    return isinstance(value, str) and value.endswith(".json")


def _is_full_matrix_json(path):
    """Check if a JSON file contains a full matrix (from eigenvalue removal)."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("type") == "full_matrix"
    except (json.JSONDecodeError, FileNotFoundError, TypeError):
        return False


def load_full_matrix(path):
    """Load a full W_res matrix from JSON. Returns (np.ndarray, metadata_dict)."""
    with open(path, "r") as f:
        data = json.load(f)
    W_res = np.array(data["W_res"], dtype=np.float64)
    metadata = data.get("metadata", {})
    return W_res, metadata


def _node_to_json(node):
    """Convert a node (int or tuple) to a JSON-serializable form."""
    if isinstance(node, tuple):
        return list(node)
    return node


def _node_from_json(node):
    """Convert a JSON-deserialized node back to its original type."""
    if isinstance(node, list):
        return tuple(node)
    return node


class Matrix:
    def __init__(self, conf):
        self.size = conf["size"]
        self.W_in_args = conf["W_in_args"]
        self.W_res_args = conf["W_res_args"]
        self.W_back_args = conf.get("W_back_args")
        self.W_in = self._init_W_in()
        self.W_res = self._init_W_res()
        self.W_back = self._init_W_back()

    def _init_W_back(self):
        if self.W_back_args is None:
            return None
        low, high = self.W_back_args["range"]
        return torch.FloatTensor(self.size).uniform_(low, high)

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
        # Standalone dense Haar-random orthogonal matrix (no lattice structure)
        if self.W_res_args["type"] == "orthogonal":
            Q, R = np.linalg.qr(np.random.randn(self.size, self.size))
            Q *= np.sign(np.diag(R))  # Haar-uniform correction
            return torch.FloatTensor(Q.astype(np.float32))

        alt_proj_iters = self.W_res_args.get("alternating_proj", 0)

        if self.W_res_args["type"] != "baseline-esn":
            m = int(sqrt(self.size))
            n = int(sqrt(self.size))

            wd = self.W_res_args["type"]
            skip_self = False
            if _is_tile_path(wd) and _is_full_matrix_json(wd):
                W_res_np, _ = load_full_matrix(wd)
                if alt_proj_iters > 0:
                    W_res_np = alternating_projections(W_res_np.astype(np.float32), alt_proj_iters)
                return torch.FloatTensor(W_res_np)
            elif _is_tile_path(wd):
                tile_data, tile_meta = load_tile_with_metadata(wd)
                tile_rows, tile_cols = tile_meta["tile_shape"]
                skip_self = tile_meta.get("optimize_self_connections", False)
                if isinstance(tile_data, OffsetTile):
                    self.G_res = _tile_offsets_to_lattice(
                        tile_data.offset_weights, tile_rows, tile_cols, m, n)
                else:
                    orth_tile = self.W_res_args.get("orthogonal_tile", False)
                    if orth_tile:
                        orth_iters = self.W_res_args.get("orthogonal_tile_iters", 50)
                        tile_data = orthogonalize_tile(tile_data, method=orth_tile, iterations=orth_iters)
                    self.G_res = self._tile_from_graph(tile_data, m, n, tile_rows, tile_cols)
            elif wd == "tile":
                self.G_res = self.tiled_rectangular(m, n)
            else:
                self.G_res = self.rectangular(m, n, neighborhood=self.W_res_args["neighborhood"])
                self.G_res = self._make_weights_negative(self.G_res, self.W_res_args["sign_frac"])
                self.G_res = self._make_graph_directed(self.G_res, self.W_res_args["directed_edges_fraction"], self.W_res_args["directed_edges_weights"])
            if not skip_self:
                self._self_connection(self.G_res)
            W_res = nx.to_numpy_array(self.G_res)
        else:
            #W_res = (torch.randint(0, 2, (self.size, self.size), dtype=torch.float32) * 2) - 1
            W_res = torch.rand(self.size, self.size) * 2 - 1
            W_res[torch.rand(self.size, self.size) > 0.1] = 0.0
            W_res[torch.eye(self.size) == 1] = self.W_res_args["self_connection"]
            W_res = W_res.numpy()

        if alt_proj_iters > 0:
            W_res = alternating_projections(W_res.astype(np.float32), alt_proj_iters)

        orthogonal_proj = self.W_res_args.get("orthogonal_proj", False)
        if orthogonal_proj == "exact":
            U, _, Vh = np.linalg.svd(W_res.astype(np.float32), full_matrices=True)
            W_res = (U @ Vh).astype(np.float32)

        return torch.FloatTensor(W_res)

    def tiled_rectangular(self, m, n):
        tile_conf = self.W_res_args["tile"]
        tile_rows, tile_cols = tile_conf["shape"]
        neighborhood = self.W_res_args.get("neighborhood", "Von_Neumann")

        # Build periodic tile so all possible neighbor offsets have defined weights
        tile_G = self.tetragonal([tile_rows, tile_cols], periodic=True, neighborhood=neighborhood)
        if tile_conf.get("method") == "random":
            for u, v, d in tile_G.edges(data=True):
                d['weight'] = np.random.random()

        # Apply sign_frac and directed edges to the tile so both patterns repeat
        self._make_weights_negative(tile_G, self.W_res_args["sign_frac"])
        tile_G = self._make_graph_directed(tile_G, self.W_res_args["directed_edges_fraction"], self.W_res_args["directed_edges_weights"])

        # Tile orthogonalization (before tiling)
        orth_tile = self.W_res_args.get("orthogonal_tile", False)
        if orth_tile:
            orth_iters = self.W_res_args.get("orthogonal_tile_iters", 50)
            tile_G = orthogonalize_tile(tile_G, method=orth_tile, iterations=orth_iters)

        # "exact" orthogonalization makes the tile dense with new edges,
        # so use offset-based tiling to preserve them
        if orth_tile == "exact":
            return _tile_edges_to_lattice_by_offset(tile_G, tile_rows, tile_cols, m, n)

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

    def _tile_from_graph(self, tile_G, m, n, tile_rows, tile_cols):
        """Tile a pre-built directed tile graph onto the full lattice."""
        neighborhood = self.W_res_args.get("neighborhood", 2)
        return _tile_edges_to_lattice(tile_G, tile_rows, tile_cols, m, n, neighborhood)

    @classmethod
    def from_tile(cls, tile_G, W_args, skip_self_connection=False):
        """Build a Matrix from a tile (nx.DiGraph or OffsetTile).

        Bypasses the normal __init__ to directly tile the given graph onto the
        full reservoir lattice. sign_frac/directed_edges are skipped since the
        tile already has them baked in.
        """
        obj = cls.__new__(cls)
        obj.size = W_args["size"]
        obj.W_in_args = W_args["W_in_args"]
        obj.W_res_args = W_args["W_res_args"]
        obj.W_back_args = W_args.get("W_back_args")
        obj.W_in = obj._init_W_in()
        obj.W_back = obj._init_W_back()

        m = int(sqrt(obj.size))
        n = int(sqrt(obj.size))

        if isinstance(tile_G, OffsetTile):
            tile_rows, tile_cols = tile_G.tile_shape
            obj.G_res = _tile_offsets_to_lattice(
                tile_G.offset_weights, tile_rows, tile_cols, m, n)
        else:
            nodes = list(tile_G.nodes())
            tile_rows = max(nd[0] for nd in nodes) + 1
            tile_cols = max(nd[1] for nd in nodes) + 1

            # Tile orthogonalization (before tiling)
            orth_tile = obj.W_res_args.get("orthogonal_tile", False)
            if orth_tile:
                orth_iters = obj.W_res_args.get("orthogonal_tile_iters", 50)
                tile_G = orthogonalize_tile(tile_G, method=orth_tile, iterations=orth_iters)

            obj.G_res = obj._tile_from_graph(tile_G, m, n, tile_rows, tile_cols)

        if not skip_self_connection:
            obj._self_connection(obj.G_res)
        W_res = nx.to_numpy_array(obj.G_res)

        # Full-matrix alternating projections (after tiling)
        alt_proj_iters = obj.W_res_args.get("alternating_proj", 0)
        if alt_proj_iters > 0:
            W_res = alternating_projections(W_res.astype(np.float32), alt_proj_iters)

        orthogonal_proj = obj.W_res_args.get("orthogonal_proj", False)
        if orthogonal_proj == "exact":
            U, _, Vh = np.linalg.svd(W_res.astype(np.float32), full_matrices=True)
            W_res = (U @ Vh).astype(np.float32)

        obj.W_res = torch.FloatTensor(W_res)

        return obj


    def tetragonal(self, dim, periodic=False, neighborhood="Von_Neumann", dist_function=None):
        m, n = dim
        G = nx.Graph()
        for i in range(m):
            for j in range(n):
                G.add_node((i, j))

        # Resolve neighborhood to a list of (di, dj) offsets
        if neighborhood == "Von_Neumann":
            offsets = self._shell_offsets(1)
        elif neighborhood == "Moore":
            offsets = self._shell_offsets(2)
        elif isinstance(neighborhood, int) and neighborhood >= 1:
            offsets = self._shell_offsets(neighborhood)
        else:
            raise ValueError(f"Unknown neighborhood: {neighborhood}. Use 'Von_Neumann', 'Moore', or an integer radius >= 1.")

        for i in range(m):
            for j in range(n):
                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if periodic:
                        G.add_edge((i, j), (ni % m, nj % n))
                    elif 0 <= ni < m and 0 <= nj < n:
                        G.add_edge((i, j), (ni, nj))

        pos = dict(zip(G, G))
        nx.set_node_attributes(G, pos, 'pos')

        return G

    @staticmethod
    def _shell_offsets(num_shells):
        """Return neighbor offsets for the first `num_shells` distance shells.

        Shells are ranked by Euclidean distance from the origin:
          shell 1: distance 1   — (±1,0), (0,±1)           — 4 offsets  (Von Neumann)
          shell 2: distance √2  — (±1,±1)                   — 4 offsets  (+ diagonals = Moore)
          shell 3: distance 2   — (±2,0), (0,±2)            — 4 offsets
          shell 4: distance √5  — (±1,±2), (±2,±1)          — 8 offsets
          shell 5: distance 2√2 — (±2,±2)                   — 4 offsets
          ...
        """
        # Generate candidate offsets up to a generous bound
        bound = num_shells + 1
        candidates = {}
        while True:
            candidates.clear()
            for di in range(-bound, bound + 1):
                for dj in range(-bound, bound + 1):
                    if (di, dj) == (0, 0):
                        continue
                    dist_sq = di * di + dj * dj
                    candidates.setdefault(dist_sq, []).append((di, dj))
            if len(candidates) >= num_shells:
                break
            bound += 1

        offsets = []
        for dist_sq in sorted(candidates)[:num_shells]:
            offsets.extend(candidates[dist_sq])
        return offsets

    def rectangular(self, m, n, periodic=False, neighborhood="Von_Neumann", dist_function=None):
        G = self.tetragonal([m, n], periodic=periodic, neighborhood=neighborhood)

        for n in G:
            pos = G.nodes[n]['pos']
            G.nodes[n]['pos'] = (pos[0], pos[1])

        for u, v, d in G.edges(data=True):
            if self.W_res_args["type"] == "constant":
                d['weight'] = 1
            elif self.W_res_args["type"] == "random":
                d['weight'] = np.random.random()
            elif self.W_res_args["type"] == "custom":
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

