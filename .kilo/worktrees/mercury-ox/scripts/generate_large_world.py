import networkx as nx
import scipy.io
import os
import numpy as np

def generate_large_scale_free(idx, n):
    print(f"Generating Large Scale-Free Graph #{idx} (V={n})...")
    m = 25 # parameter for Barabasi-Albert
    G = nx.barabasi_albert_graph(n, m)
    adj = nx.adjacency_matrix(G).astype(float)
    adj.data = np.random.uniform(1.0, 10.0, size=adj.nnz)
    
    out_dir = "data/large_world"
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{out_dir}/large_sf_{idx}_V{n}_E{adj.nnz}.mtx"
    scipy.io.mmwrite(filename, adj)
    print(f"  -> Saved to {filename}")
    return filename

if __name__ == "__main__":
    sizes = [100000, 150000, 200000, 250000]
    for i, s in enumerate(sizes):
        generate_large_scale_free(i+1, s)
    print("Done generating 4 large world graphs!")
