import numpy as np
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from concurrent.futures import ProcessPoolExecutor
import functools
from collections import Counter
import graph_data

# ==========================================
# 1. INDEPENDENT WORKER FUNCTION
# ==========================================

def route_individual_pair(pair, graph_gdf_segments, angle_map, edges_dict, params, initial_pheromones):
    source_node, machine_node = pair

    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)
    gamma = params.get('gamma', 0.5)
    n_ants = params.get('n_ants', 70)
    n_iterations = params.get('n_iterations', 400)

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
                    idx = path.index(next_node)
                    path = path[:idx + 1]
                    current = next_node
                    prev = path[-2] if len(path) > 1 else None
                else:
                    path.append(next_node)
                    if next_node == source_node:
                        iteration_paths.append(path)
                        break
                    prev = current
                    current = next_node

        for e in local_pheromones: 
            local_pheromones[e] *= 0.9
        for p in iteration_paths:
            d = sum(edges_dict[tuple(sorted((p[i], p[i+1])))] for i in range(len(p)-1))
            if d < best_dist:
                best_dist = d
                best_path = p
            for i in range(len(p)-1):
                local_pheromones[tuple(sorted((p[i], p[i+1])))] += (100.0 / d)

    return pair, best_path, best_dist


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def evaluate_performance(all_routes, edges_dict, graph_data):
    total_cable = 0
    unique_edges = set()
    for path in all_routes.values():
        if not path: continue
        total_cable += sum(edges_dict[tuple(sorted((path[i], path[i+1])))] for i in range(len(path)-1))
        for i in range(len(path)-1): 
            unique_edges.add(tuple(sorted((path[i], path[i+1]))))
    
    return {
        "Total_Cable_m": round(total_cable, 2), 
        "Trench_m": round(sum(edges_dict[e] for e in unique_edges), 2)
    }


def plot_network(graph_data, all_routes, title="Network Layout", line_color='red'):
    edge_counts = Counter()
    for path in all_routes.values():
        for i in range(len(path)-1): 
            edge_counts[tuple(sorted((path[i], path[i+1])))] += 1

    graph_data.gdf_segments['cable_count'] = graph_data.gdf_segments.apply(
        lambda r: edge_counts.get(tuple(sorted((int(r['i']), int(r['j'])))), 0), axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot base network in light gray
    graph_data.gdf_segments.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5, zorder=1)
    
    # Plot used segments
    used = graph_data.gdf_segments[graph_data.gdf_segments['cable_count'] > 0]
    if not used.empty:
        max_cables = used['cable_count'].max() or 1
        used.plot(ax=ax, color=line_color, linewidth=1 + (used['cable_count'] / max_cables * 5), zorder=2)
        
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()


# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    file_path = "Capstone_M2DS_26/dataclass/graph_data_exemple.pkl"
    with open(file_path, "rb") as f: 
        graph_data1 = pickle.load(f)

    # Build Initial Graph
    G_full = nx.Graph()
    for _, r in graph_data1.gdf_segments.iterrows():
        G_full.add_edge(int(r['i']), int(r['j']), weight=r['length_m'])

    # --- KEEP ONLY LARGEST CONNECTED COMPONENT ---
    print(f"🔍 Original graph has {G_full.number_of_nodes()} nodes.")
    largest_cc_nodes = max(nx.connected_components(G_full), key=len)
    G = G_full.subgraph(largest_cc_nodes).copy()
    print(f"✅ Filtered to Largest Connected Component: {G.number_of_nodes()} nodes.")

    edges_dict = {tuple(sorted(e)): d['weight'] for *e, d in G.edges(data=True)}

    # Categorize connection pairs
    raw_pairs = [(int(r['tenant']), int(r['aboutissant'])) for _, r in graph_data1.df_cables.dropna().iterrows()]
    connection_pairs = []
    failed_pairs = []

    for src, tgt in raw_pairs:
        if src in largest_cc_nodes and tgt in largest_cc_nodes:
            connection_pairs.append((src, tgt))
        else:
            failed_pairs.append((src, tgt))

    print(f"📉 Connection pairs adjusted: {len(raw_pairs)} -> {len(connection_pairs)} ({len(failed_pairs)} unreachable pairs removed)")

    # --- ISOLATION ANALYSIS (DISTANCE TO MAIN NETWORK) ---
    if failed_pairs:
        print("\n📏 Analyzing Isolated Nodes...")
        # Get all unique nodes that were excluded
        isolated_nodes = set()
        for src, tgt in failed_pairs:
            if src not in largest_cc_nodes: isolated_nodes.add(src)
            if tgt not in largest_cc_nodes: isolated_nodes.add(tgt)
        
        # Prepare geometries
        nodes_idx = graph_data1.gdf_nodes.set_index('node_id')
        lcc_geoms = nodes_idx.loc[nodes_idx.index.intersection(list(largest_cc_nodes)), 'geometry']
        
        distances = []
        for node in isolated_nodes:
            if node in nodes_idx.index:
                point = nodes_idx.loc[node, 'geometry']
                # Calculate minimum distance to any point in the main network
                min_dist = lcc_geoms.distance(point).min()
                distances.append(min_dist)
        
        if distances:
            print(f"⚠️ Found {len(isolated_nodes)} isolated machines/sources.")
            print(f"   ➔ Minimum gap to bridge: {min(distances):.2f} meters")
            print(f"   ➔ Maximum gap to bridge: {max(distances):.2f} meters")
            print(f"   ➔ Average gap to bridge: {np.mean(distances):.2f} meters")

    # Filter angle map for valid nodes
    angle_map = {(int(r['i']), int(r['j']), int(r['k'])): r['abs_cosine'] 
                 for _, r in graph_data1.df_angles.iterrows() 
                 if all(n in largest_cc_nodes for n in [int(r['i']), int(r['j']), int(r['k'])])}

    # --- DIJKSTRA INIT ---
    print("\n🧠 Computing Dijkstra Shortest Paths & Initializing Pheromones...")
    dijkstra_routes = {}
    global_init_pheromones = {e: 1.0 for e in edges_dict.keys()}
    
    for src, tgt in connection_pairs:
        try:
            d_path = nx.shortest_path(G, source=src, target=tgt, weight='weight')
            dijkstra_routes[(src, tgt)] = d_path
            for i in range(len(d_path)-1):
                edge = tuple(sorted((d_path[i], d_path[i+1])))
                global_init_pheromones[edge] += 15.0
        except nx.NetworkXNoPath: 
            continue

    # Evaluate and Plot Dijkstra Baseline
    print("\n--- BASELINE: DIJKSTRA SHORTEST PATHS ---")
    dijkstra_metrics = evaluate_performance(dijkstra_routes, edges_dict, graph_data1)
    print(f"Result: {len(dijkstra_routes)}/{len(connection_pairs)} pairs connected.")
    print(dijkstra_metrics)
    plot_network(graph_data1, dijkstra_routes, title="Baseline: Dijkstra Absolute Shortest Paths", line_color='blue')

    # --- ACO EXECUTION ---
    print("\n🐜 Starting Ant Colony Optimization (ACO)...")
    params = {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.5, 'n_ants': 70, 'n_iterations': 10}
    worker_task = functools.partial(route_individual_pair, graph_gdf_segments=graph_data1.gdf_segments,
                                    angle_map=angle_map, edges_dict=edges_dict, params=params, 
                                    initial_pheromones=global_init_pheromones)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker_task, connection_pairs))

    all_routes = {p: path for p, path, d in results if path}
    
    print("\n--- FINAL RESULTS: ACO OPTIMIZED PATHS ---")
    print(f"Result: {len(all_routes)}/{len(connection_pairs)} pairs connected in the main component.")
    aco_metrics = evaluate_performance(all_routes, edges_dict, graph_data1)
    print(aco_metrics)
    
    if dijkstra_metrics["Trench_m"] > 0:
        savings = dijkstra_metrics["Trench_m"] - aco_metrics["Trench_m"]
        print(f"💡 Trench Savings vs Dijkstra: {savings:.2f} meters")

    plot_network(graph_data1, all_routes, title="Optimized Network (Dijkstra-Initialized ACO)", line_color='red')