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
# 1. INDEPENDENT WORKER FUNCTION (For Parallelism)
# ==========================================
def route_individual_pair(pair, graph_gdf_segments, angle_map, edges_dict, params):
    """
    This function runs on a single CPU core. 
    It finds the best path for ONE pair from scratch to avoid pheromone bias.
    """
    source_node, machine_node = pair
    
    # Hyperparameters
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 2.0)
    gamma = params.get('gamma', 1.5)
    n_ants = params.get('n_ants', 15)
    n_iterations = params.get('n_iterations', 10)
    
    # Local Pheromone Map for this specific pair
    local_pheromones = {edge: 1.0 for edge in edges_dict.keys()}
    
    best_path = None
    best_dist = float('inf')

    for _ in range(n_iterations):
        iteration_paths = []
        for _ in range(n_ants):
            # Construct path (Machine -> Source)
            path = [machine_node]
            prev = None
            current = machine_node
            
            # Max steps to prevent infinite loops
            for _ in range((len(edges_dict) // 2)*10):
                # Get neighbors
                neighbors = []
                for edge, dist in edges_dict.items():
                    if current in edge:
                        neighbor = edge[0] if edge[1] == current else edge[1]
                        neighbors.append((neighbor, dist))
                
                if not neighbors: break
                
                # Filter out immediate back-tracking
                candidates = [n for n in neighbors if n[0] != prev]
                if not candidates: candidates = neighbors
                
                # Calculate Probabilities (Vectorized-style)
                probs = []
                for neighbor, dist in candidates:
                    edge = tuple(sorted((current, neighbor)))
                    tau = local_pheromones[edge] ** alpha
                    eta = (1.0 / dist) ** beta
                    
                    angle_factor = 1.0
                    if prev is not None:
                        cosine = angle_map.get((prev, current, neighbor), 0.5)
                        angle_factor = (cosine + 0.1) ** gamma
                    
                    probs.append(tau * eta * angle_factor)
                
                sum_p = sum(probs)
                if sum_p == 0:
                    next_node = random.choice(candidates)[0]
                else:
                    next_node = np.random.choice([c[0] for c in candidates], p=[p/sum_p for p in probs])
                
                # Loop Erasure
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
        
        # Update Local Pheromones (Evaporation + Deposition)
        for e in local_pheromones: local_pheromones[e] *= 0.9
        for p in iteration_paths:
            d = sum(edges_dict[tuple(sorted((p[i], p[i+1])))] for i in range(len(p)-1))
            if d < best_dist:
                best_dist = d
                best_path = p
            for i in range(len(p)-1):
                local_pheromones[tuple(sorted((p[i], p[i+1])))] += (100.0 / d)

    return pair, best_path, best_dist

# ==========================================
# 2. EVALUATION & VISUALIZATION FUNCTIONS
# ==========================================
def evaluate_performance(all_routes, edges_dict, angle_map, graph_data):
    total_cable = 0
    unique_edges = set()
    sinuosity = []
    
    nodes_idx = graph_data.gdf_nodes.set_index('node_id')

    for (src, tgt), path in all_routes.items():
        if not path: continue
        
        # Path Length
        d_actual = 0
        for i in range(len(path)-1):
            edge = tuple(sorted((path[i], path[i+1])))
            d_actual += edges_dict[edge]
            unique_edges.add(edge)
        
        total_cable += d_actual
        
        # Sinuosity
        p1 = nodes_idx.loc[path[0], 'geometry']
        p2 = nodes_idx.loc[path[-1], 'geometry']
        d_direct = p1.distance(p2)
        if d_direct > 0: sinuosity.append(d_actual / d_direct)

    trench_len = sum(edges_dict[e] for e in unique_edges)
    
    return {
        "Total_Cable_Used": round(total_cable, 2),
        "Trench_Footprint": round(trench_len, 2),
        "Avg_Sinuosity": round(np.mean(sinuosity), 3) if sinuosity else 0
    }


def plot_final_network(graph_data1, all_routes, connection_pairs):
    """
    Plots the final network, showing line thickness based on how many 
    cables pass through a specific trench.
    """
    # 1. Count how many times each edge is used across all routes
    edge_counts = Counter()
    for path in all_routes.values():
        if not path or len(path) < 2:
            continue
        for i in range(len(path) - 1):
            edge = tuple(sorted((path[i], path[i+1])))
            edge_counts[edge] += 1

    # 2. Map these counts back to the GeoDataFrame
    counts_list = []
    for _, row in graph_data1.gdf_segments.iterrows():
        edge = tuple(sorted((int(row['i']), int(row['j']))))
        counts_list.append(edge_counts.get(edge, 0))
    
    graph_data1.gdf_segments['cable_count'] = counts_list

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Base Network (Unused trenches in light gray)
    base_mask = graph_data1.gdf_segments['cable_count'] == 0
    graph_data1.gdf_segments[base_mask].plot(ax=ax, color='lightgray', linewidth=0.6, alpha=0.5, zorder=1)

    # Active Network (Red lines, thickness = number of cables)
    used_mask = graph_data1.gdf_segments['cable_count'] > 0
    if used_mask.any():
        used_edges = graph_data1.gdf_segments[used_mask].copy()
        # Scale linewidth based on density
        max_cables = used_edges['cable_count'].max()
        used_edges['lw'] = 1.0 + (used_edges['cable_count'] / max_cables) * 4.0
        used_edges.plot(ax=ax, color='red', linewidth=used_edges['lw'], alpha=0.8, label='Active Trenches', zorder=2)

    # 4. Plot Source and Machine Nodes
    sources_ids = list(set([p[0] for p in connection_pairs]))
    targets_ids = list(set([p[1] for p in connection_pairs]))

    gdf_nodes_idx = graph_data1.gdf_nodes.set_index('node_id')

    # Sources (Green Triangles)
    src_geom = gdf_nodes_idx.loc[gdf_nodes_idx.index.intersection(sources_ids), 'geometry']
    if not src_geom.empty:
        ax.scatter(src_geom.x, src_geom.y, color='lime', s=150, marker='^', edgecolors='black', label='Sources', zorder=4)

    # Machines (Blue Dots)
    tgt_geom = gdf_nodes_idx.loc[gdf_nodes_idx.index.intersection(targets_ids), 'geometry']
    if not tgt_geom.empty:
        ax.scatter(tgt_geom.x, tgt_geom.y, color='blue', s=50, marker='o', edgecolors='white', label='Machines', zorder=3)

    plt.title("Final Network Routing Density", fontsize=16, fontweight='bold')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # Handle Legend avoiding duplicate labels from geopandas
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig("Final_Network_Heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Load Data
    file_path = "Capstone_M2DS_26/dataclass/graph_data_exemple.pkl"
    # Make sure to replace this with your actual path if running outside your current directory
    with open(file_path, "rb") as f:
        graph_data1 = pickle.load(f)
    # Build shared data structures

    edges_dict = {}
    for _, row in graph_data1.gdf_segments.iterrows():
        edges_dict[tuple(sorted((int(row['i']), int(row['j']))))] = row['length_m']
        
    angle_map = {}
    for _, row in graph_data1.df_angles.iterrows():
        angle_map[(int(row['i']), int(row['j']), int(row['k']))] = row['abs_cosine']
        angle_map[(int(row['k']), int(row['j']), int(row['i']))] = row['abs_cosine']

    # Prep Pairs
    # (Assuming graph cleaning logic from your original code is applied here)
    connection_pairs = [(int(row['tenant']), int(row['aboutissant'])) 
                        for _, row in graph_data1.df_cables.dropna().iterrows()]

    # Global Parameters
    params = {'alpha': 1.0, 'beta': 3.0, 'gamma': 2.0, 'n_ants': 20, 'n_iterations': 15}

    print(f"🚀 Starting Parallel Routing for {len(connection_pairs)} pairs...")

    # EXECUTE IN PARALLEL
    all_routes = {}
    worker_task = functools.partial(
        route_individual_pair, 
        graph_gdf_segments=graph_data1.gdf_segments,
        angle_map=angle_map,
        edges_dict=edges_dict,
        params=params
    )

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker_task, connection_pairs))

    for pair, path, dist in results:
        if path:
            all_routes[pair] = path

    # Results
    metrics = evaluate_performance(all_routes, edges_dict, angle_map, graph_data1)
    print("\n--- FINAL RESULTS ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Visualizing the final output
    print("\n🗺️ Generating the final network density plot...")
    plot_final_network(graph_data1, all_routes, connection_pairs)
