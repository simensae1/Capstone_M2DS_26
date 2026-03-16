import numpy as np
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import functools
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import graph_data

# ==========================================
# 1. INDEPENDENT WORKER FUNCTION
# ==========================================

def route_individual_pair(pair, graph_gdf_segments, angle_map, edges_dict, params, initial_pheromones):
    source_node, machine_node = pair
    alpha, beta, gamma = params.get('alpha', 1.0), params.get('beta', 1.0), params.get('gamma', 0.5)
    n_ants, n_iterations = params.get('n_ants', 70), params.get('n_iterations', 400)

    local_pheromones = initial_pheromones.copy()
    best_path, best_dist = None, float('inf')

    for _ in range(n_iterations):
        iteration_paths = []
        for _ in range(n_ants):
            path, prev, current = [machine_node], None, machine_node
            # Limit steps to avoid infinite loops in complex graphs
            for _ in range(len(edges_dict) // 2):
                neighbors = [(e[0] if e[1] == current else e[1], d) for e, d in edges_dict.items() if current in e]
                if not neighbors: break
                candidates = [n for n in neighbors if n[0] != prev] or neighbors
                
                probs = []
                for neighbor, dist in candidates:
                    edge = tuple(sorted((current, neighbor)))
                    tau = local_pheromones.get(edge, 1.0) ** alpha
                    eta = (1.0 / dist) ** beta
                    angle_f = (angle_map.get((prev, current, neighbor), 0.5) + 0.1) ** gamma if prev else 1.0
                    probs.append(tau * eta * angle_f)
                
                sum_p = sum(probs)
                next_node = random.choice(candidates)[0] if sum_p == 0 else \
                            np.random.choice([c[0] for c in candidates], p=[p/sum_p for p in probs])

                if next_node in path:
                    idx = path.index(next_node)
                    path = path[:idx + 1]
                    current = next_node
                    prev = path[-2] if len(path) > 1 else None
                else:
                    path.append(next_node)
                    if next_node == source_node:
                        iteration_paths.append(path); break
                    prev, current = current, next_node

        for e in local_pheromones: local_pheromones[e] *= 0.9
        for p in iteration_paths:
            d = sum(edges_dict[tuple(sorted((p[i], p[i+1])))] for i in range(len(p)-1))
            if d < best_dist: best_dist, best_path = d, p
            for i in range(len(p)-1):
                local_pheromones[tuple(sorted((p[i], p[i+1])))] += (100.0 / d)

    return pair, best_path, best_dist

# ==========================================
# 2. ENHANCED CONNECTIVITY REPAIR
# ==========================================

def repair_graph_connectivity(G, edges_dict, connection_pairs, nodes_gdf):
    nodes_idx = nodes_gdf.set_index('node_id')
    lcc_nodes = max(nx.connected_components(G), key=len)
    
    # Identify unique nodes needed for connections
    required_nodes = set()
    for src, tgt in connection_pairs:
        required_nodes.add(src)
        required_nodes.add(tgt)
    
    # Check for nodes missing from geometry entirely
    missing_geo = [n for n in required_nodes if n not in nodes_idx.index]
    if missing_geo:
        print(f"❌ CRITICAL: {len(missing_geo)} nodes have NO geometry in gdf_nodes (e.g., {missing_geo[:3]})")

    isolated_nodes = [n for n in required_nodes if n not in lcc_nodes and n in nodes_idx.index]
    
    if not isolated_nodes:
        print("✅ No isolated nodes found that have geometry.")
        return G, edges_dict, lcc_nodes

    print(f"🛠️ Repairing {len(isolated_nodes)} isolated nodes by snapping to nearest LCC node...")
    lcc_list = list(lcc_nodes)
    lcc_geoms = nodes_idx.loc[lcc_list, 'geometry']
    
    new_G = G.copy()
    new_edges = edges_dict.copy()

    for node in isolated_nodes:
        point = nodes_idx.loc[node, 'geometry']
        distances = lcc_geoms.distance(point)
        nearest_node_id = distances.idxmin()
        dist_val = max(distances.min(), 0.1)
        
        edge = tuple(sorted((node, nearest_node_id)))
        new_G.add_edge(*edge, weight=dist_val)
        new_edges[edge] = dist_val
        
    final_lcc = max(nx.connected_components(new_G), key=len)
    return new_G, new_edges, final_lcc


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
    safe_title = title.replace(" ", "_").replace(":", "")
    plt.savefig(f"{safe_title}.png", dpi=300, bbox_inches='tight')
    plt.show()


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

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    with open("Capstone_M2DS_26/dataclass/graph_data_exemple.pkl", "rb") as f:
        graph_data1 = pickle.load(f)

    # Initial graph setup
    G_init = nx.Graph()
    for _, r in graph_data1.gdf_segments.iterrows():
        G_init.add_edge(int(r['i']), int(r['j']), weight=r['length_m'])
    
    edges_dict_raw = {tuple(sorted((int(r['i']), int(r['j'])))): r['length_m'] for _, r in graph_data1.gdf_segments.iterrows()}
    raw_pairs = [(int(r['tenant']), int(r['aboutissant'])) for _, r in graph_data1.df_cables.dropna().iterrows()]

    # REPAIR
    G_repaired, edges_dict, final_lcc = repair_graph_connectivity(G_init, edges_dict_raw, raw_pairs, graph_data1.gdf_nodes)
    
    # DIJKSTRA
    dijkstra_routes = {}
    global_init_pheromones = {e: 1.0 for e in edges_dict.keys()}
    
    # Check why they fail inside the loop
    connectivity_results = []
    for src, tgt in raw_pairs:
        try:
            d_path = nx.shortest_path(G_repaired, source=src, target=tgt, weight='weight')
            dijkstra_routes[(src, tgt)] = d_path
            connectivity_results.append(True)
            for i in range(len(d_path)-1):
                global_init_pheromones[tuple(sorted((d_path[i], d_path[i+1])))] += 15.0
        except Exception:
            connectivity_results.append(False)

    print(f"\n--- DIJKSTRA RESULTS ---")
    print(f"Total Pairs: {len(raw_pairs)}")
    print(f"Successfully Connected by Dijkstra: {sum(connectivity_results)}")
    plot_network(graph_data1, dijkstra_routes, title="DIJKSTRA RESULTS", line_color='red')
    if sum(connectivity_results) < len(raw_pairs):
        print("⚠️  Dijkstra still cannot reach all nodes. Possible reasons:")
        print("   1. Some nodes are completely missing from the 'gdf_nodes' coordinate table.")
        print("   2. The Source node is on one island and the Machine is on another (repair only snaps to the LARGE component).")

    # ACO
    print("\n🐜 Starting ACO...")
    connection_pairs = [k for k in dijkstra_routes.keys()]
    angle_map = {(int(r['i']), int(r['j']), int(r['k'])): r['abs_cosine'] for _, r in graph_data1.df_angles.iterrows()}

    with ProcessPoolExecutor() as executor:
        worker_task = functools.partial(route_individual_pair, graph_gdf_segments=graph_data1.gdf_segments,
                                        angle_map=angle_map, edges_dict=edges_dict, params={'n_ants':70, 'n_iterations':400}, 
                                        initial_pheromones=global_init_pheromones)
        results = list(executor.map(worker_task, connection_pairs))

    all_routes = {p: path for p, path, d in results if path}
    print(f"✅ Final ACO Result: {len(all_routes)}/{len(raw_pairs)} pairs connected.")
    evaluate_performance(all_routes, edges_dict, angle_map, graph_data1)
    plot_network(graph_data1, all_routes, title="AOC RESULTS", line_color='red')

