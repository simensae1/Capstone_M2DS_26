import numpy as np
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from concurrent.futures import ProcessPoolExecutor
import functools
from IPython.display import clear_output
from collections import Counter
import graph_data

# ==========================================
# 1. INDEPENDENT WORKER FUNCTION
# ==========================================


def route_individual_pair(pair, graph_gdf_segments, angle_map, edges_dict, params, initial_pheromones):
    source_node, machine_node = pair

    alpha = params.get('alpha', 1.2)
    beta = params.get('beta', 3.0)
    gamma = params.get('gamma', 2.5)
    n_ants = params.get('n_ants', 20)
    n_iterations = params.get('n_iterations', 20)

    local_pheromones = initial_pheromones.copy()
    best_path = None
    best_dist = float('inf')

    for _ in range(n_iterations):
        iteration_paths = []
        for _ in range(n_ants):
            path = [machine_node]
            prev = None
            current = machine_node

            for _ in range(len(edges_dict)):
                neighbors = []
                for edge, dist in edges_dict.items():
                    if current in edge:
                        neighbor = edge[0] if edge[1] == current else edge[1]
                        neighbors.append((neighbor, dist))

                if not neighbors:
                    break
                candidates = [n for n in neighbors if n[0] != prev]
                if not candidates:
                    candidates = neighbors

                probs = []
                for neighbor, dist in candidates:
                    edge = tuple(sorted((current, neighbor)))
                    tau = local_pheromones.get(edge, 1.0) ** alpha
                    eta = (1.0 / dist) ** beta
                    angle_factor = (angle_map.get((prev, current, neighbor), 0.5) + 0.1) ** gamma if prev else 1.0
                    probs.append(tau * eta * angle_factor)

                sum_p = sum(probs)
                if sum_p == 0:
                    next_node = random.choice(candidates)[0]
                else:
                    next_node = np.random.choice([c[0] for c in candidates], p=[p/sum_p for p in probs])

                if next_node in path:
                    idx = path.index(next_node); 
                    path = path[:idx + 1]
                    current = next_node
                    prev = path[-2] if len(path) > 1 else None
                else:
                    path.append(next_node)
                    if next_node == source_node:
                        iteration_paths.append(path); break
                    prev = current
                    current = next_node

        for e in local_pheromones: 
            local_pheromones[e] *= 0.9
        for p in iteration_paths:
            d = sum(edges_dict[tuple(sorted((p[i], p[i+1])))] for i in range(len(p)-1))
            if d < best_dist:
                best_dist = d; best_path = p
            for i in range(len(p)-1):
                local_pheromones[tuple(sorted((p[i], p[i+1])))] += (100.0 / d)

    return pair, best_path, best_dist

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================


def evaluate_performance(all_routes, edges_dict, graph_data):
    total_cable = 0; unique_edges = set()
    for path in all_routes.values():
        if not path: continue
        total_cable += sum(edges_dict[tuple(sorted((path[i], path[i+1])))] for i in range(len(path)-1))
        for i in range(len(path)-1): unique_edges.add(tuple(sorted((path[i], path[i+1]))))
    return {"Total_Cable_m": round(total_cable, 2), "Trench_m": round(sum(edges_dict[e] for e in unique_edges), 2)}


def plot_final_network(graph_data, all_routes):
    edge_counts = Counter()
    for path in all_routes.values():
        for i in range(len(path)-1): 
            edge_counts[tuple(sorted((path[i], path[i+1])))] += 1

    graph_data.gdf_segments['cable_count'] = graph_data.gdf_segments.apply(
        lambda r: edge_counts.get(tuple(sorted((int(r['i']), int(r['j'])))), 0), axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    graph_data.gdf_segments.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5)
    used = graph_data.gdf_segments[graph_data.gdf_segments['cable_count'] > 0]
    if not used.empty:
        used.plot(ax=ax, color='red', linewidth=1 + (used['cable_count']/used['cable_count'].max()*5))
    plt.title("Largest Connected Component - Optimized Network")
    plt.show()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================


if __name__ == "__main__":
    file_path = "Capstone_M2DS_26/dataclass/graph_data_exemple.pkl"
    with open(file_path, "rb") as f: graph_data1 = pickle.load(f)

    # Build Initial Graph
    G_full = nx.Graph()
    for _, r in graph_data1.gdf_segments.iterrows():
        G_full.add_edge(int(r['i']), int(r['j']), weight=r['length_m'])

    # --- KEEP ONLY LARGEST CONNECTED COMPONENT ---
    print(f"🔍 Original graph has {G_full.number_of_nodes()} nodes.")
    largest_cc_nodes = max(nx.connected_components(G_full), key=len)
    G = G_full.subgraph(largest_cc_nodes).copy()
    print(f"✅ Filtered to Largest Connected Component: {G.number_of_nodes()} nodes.")

    # Filter dicts based on new graph
    edges_dict = {tuple(sorted(e)): d['weight'] for *e, d in G.edges(data=True)}

    # Filter connection pairs: only keep pairs where BOTH nodes are in the largest component
    raw_pairs = [(int(r['tenant']), int(r['aboutissant'])) for _, r in graph_data1.df_cables.dropna().iterrows()]
    connection_pairs = [p for p in raw_pairs if p[0] in largest_cc_nodes and p[1] in largest_cc_nodes]

    print(f"📉 Connection pairs adjusted: {len(raw_pairs)} -> {len(connection_pairs)} (Unreachable pairs removed)")

    # Filter angle map for valid nodes
    angle_map = {(int(r['i']), int(r['j']), int(r['k'])): r['abs_cosine'] 
                 for _, r in graph_data1.df_angles.iterrows() 
                 if all(n in largest_cc_nodes for n in [int(r['i']), int(r['j']), int(r['k'])])}

    # Dijkstra Init
    print(" Initializing Pheromones...")
    global_init_pheromones = {e: 1.0 for e in edges_dict.keys()}
    for src, tgt in connection_pairs:
        try:
            d_path = nx.shortest_path(G, source=src, target=tgt, weight='weight')
            for i in range(len(d_path)-1):
                edge = tuple(sorted((d_path[i], d_path[i+1])))
                global_init_pheromones[edge] += 5.0
        except nx.NetworkXNoPath: 
            continue

    print("ACO...")
    params = {'alpha': 1.0, 'beta': 1, 'gamma': 1.0, 'n_ants': 70, 'n_iterations': 200}
    worker_task = functools.partial(route_individual_pair, graph_gdf_segments=graph_data1.gdf_segments,
                                    angle_map=angle_map, edges_dict=edges_dict, params=params, 
                                    initial_pheromones=global_init_pheromones)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker_task, connection_pairs))

    all_routes = {p: path for p, path, d in results if path}
    print(f"\n📊 Result: {len(all_routes)}/{len(connection_pairs)} pairs connected in the main component.")
    print(evaluate_performance(all_routes, edges_dict, graph_data1))
    plot_final_network(graph_data1, all_routes)
