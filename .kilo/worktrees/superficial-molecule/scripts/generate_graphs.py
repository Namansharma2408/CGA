import networkx as nx
import scipy.io
import scipy.sparse as sp
import os
import argparse
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def generate_graph(args):
    idx, type_str, n, m_edges, sub_dir = args
    G = None

    if type_str == "scale_free":
        m = max(1, m_edges // n)
        G = nx.barabasi_albert_graph(n, m)
    elif type_str == "small_world":
        k = max(2, (m_edges * 2) // n)
        G = nx.watts_strogatz_graph(n, k, 0.1)
    elif type_str == "erdos_renyi":
        p = (2.0 * m_edges) / (n * (n - 1))
        G = nx.erdos_renyi_graph(n, p)
    elif type_str == "mesh":
        dim = int(np.sqrt(n))
        G = nx.grid_2d_graph(dim, dim)
        G = nx.relabel_nodes(G, {node: i for i, node in enumerate(G.nodes())})

    if G is not None:
        adj = nx.adjacency_matrix(G).astype(float)
        adj.data = np.random.uniform(1.0, 10.0, size=adj.nnz)

        filename = f"data/{sub_dir}/graph_{idx:03d}_{type_str}_V{n}_E{adj.nnz}.mtx"
        scipy.io.mmwrite(filename, adj)
        return filename
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 400 pseudo real-world large graphs.")
    parser.add_argument("--count", type=int, default=400, help="Number of graphs to generate")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads")
    parser.add_argument("--train_only", action="store_true", help="Force all graphs into data/train/")
    parser.add_argument("--test_only", action="store_true", help="Force all graphs into data/test/")
    args = parser.parse_args()

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    print(f"Generating {args.count} real-world analog graphs...")
    tasks = []

    for i in range(args.count):
        n = random.randint(5_000, 20_000)
        edges = n * random.randint(5, 50)
        gtype = random.choice(["scale_free", "small_world", "erdos_renyi", "mesh"])

        if args.train_only: sub_dir = "train"
        elif args.test_only: sub_dir = "test"
        else: sub_dir = "train" if np.random.rand() < 0.8 else "test"

        tasks.append((i, gtype, n, edges, sub_dir))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, path in enumerate(executor.map(generate_graph, tasks)):
            if (i+1) % 20 == 0:
                print(f" -> Generated {i+1}/{args.count} datasets")

    print("Done! Datasets saved to data/train/ and data/test/")