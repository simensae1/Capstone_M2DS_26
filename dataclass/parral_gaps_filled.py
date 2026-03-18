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
import itertools
import pandas as pd

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
# 2. HELPER FUNCTIONS & REPAIR LOGIC
# ==========================================

def repair_graph_connectivity(G, edges_dict, connection_pairs, nodes_gdf):
    """
    Identifies nodes outside the LCC, logs isolation analysis (from Code 1),
    and creates virtual edges to snap them to the nearest LCC node.
    """
    nodes_idx = nodes_gdf.set_index('node_id')
    lcc_nodes = max(nx.connected_components(G), key=len)

    # Check all target nodes
    all_target_nodes = set()
    for src, tgt in connection_pairs:
        all_target_nodes.add(src)
        all_target_nodes.add(tgt)

    # Identify which nodes are floating outside the main network
    isolated_nodes = all_target_nodes - lcc_nodes
    if not isolated_nodes:
        return G, edges_dict, lcc_nodes

    print(f"\n📏 Analyzing and Repairing Connectivity for {len(isolated_nodes)} Isolated Nodes...")
    lcc_list = list(lcc_nodes)
    lcc_geoms = nodes_idx.loc[lcc_list, 'geometry']

    new_G = G.copy()
    new_edges = edges_dict.copy()

    distances_list = []

    for node in isolated_nodes:
        if node not in nodes_idx.index:
            continue

        point = nodes_idx.loc[node, 'geometry']

        # Calculate distance to main network
        distances = lcc_geoms.distance(point)
        min_dist = distances.min()
        distances_list.append(min_dist)

        nearest_node_id = distances.idxmin()
        dist_val = max(min_dist, 0.1)  # Prevent 0.0 distance routing issues

        # Add virtual bridge edge
        edge = tuple(sorted((node, nearest_node_id)))
        new_G.add_edge(*edge, weight=dist_val)
        new_edges[edge] = dist_val

    # --- ISOLATION ANALYSIS METRICS ---
    if distances_list:
        print(f"⚠️ Found {len(isolated_nodes)} isolated machines/sources.")
        print(f"   ➔ Minimum gap bridged: {min(distances_list):.2f} meters")
        print(f"   ➔ Maximum gap bridged: {max(distances_list):.2f} meters")
        print(f"   ➔ Average gap bridged: {np.mean(distances_list):.2f} meters")

    final_lcc = max(nx.connected_components(new_G), key=len)
    return new_G, new_edges, final_lcc


def evaluate_performance(all_routes, edges_dict, graph_data):
    total_cable = 0
    unique_edges = set()
    for path in all_routes.values():
        if not path:
            continue
        total_cable += sum(edges_dict[tuple(sorted((path[i], path[i+1])))] for i in range(len(path)-1))
        for i in range(len(path)-1):
            unique_edges.add(tuple(sorted((path[i], path[i+1]))))

    return {
        "Total_Cable_m": round(total_cable, 2),
        "Trench_m": round(sum(edges_dict[e] for e in unique_edges), 2)
    }


def plot_network(graph_data, all_routes, edges_dict_raw, title="Network Layout", line_color='red'):
    """
    Combined Plotting: Updates 'cable_count' on gdf_segments for real edges,
    plots 'virtual' bridges, and overlays source/machine nodes.
    """
    edge_counts = Counter()
    for path in all_routes.values():
        if not path:
            continue
        for i in range(len(path)-1): 
            edge_counts[tuple(sorted((path[i], path[i+1])))] += 1

    # Map usage to original spatial dataframe
    graph_data.gdf_segments['cable_count'] = graph_data.gdf_segments.apply(
        lambda r: edge_counts.get(tuple(sorted((int(r['i']), int(r['j'])))), 0), axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))

    # 1. Plot base network in light gray
    graph_data.gdf_segments.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5, zorder=1)

    # 2. Plot used segments
    used = graph_data.gdf_segments[graph_data.gdf_segments['cable_count'] > 0]
    max_cables = max(edge_counts.values()) if edge_counts else 1

    if not used.empty:
        used.plot(ax=ax, color=line_color, linewidth=1 + (used['cable_count'] / max_cables * 5), zorder=2)

    # 3. Plot virtual repaired edges (dotted orange lines)
    nodes_idx = graph_data.gdf_nodes.set_index('node_id')
    for (u, v), count in edge_counts.items():
        if tuple(sorted((u, v))) not in edges_dict_raw:
            if u in nodes_idx.index and v in nodes_idx.index:
                p1 = nodes_idx.loc[u, 'geometry']
                p2 = nodes_idx.loc[v, 'geometry']
                linewidth = 1 + (count / max_cables * 5)
                ax.plot([p1.x, p2.x], [p1.y, p2.y], color='orange', linestyle='--', linewidth=linewidth, zorder=3)

    # ---------------------------------------------------------
    # 4. PLOT SOURCES AND MACHINES (NEW)
    # ---------------------------------------------------------
    # Extract unique source and machine IDs from the pairs
    sources_ids = set(pair[0] for pair in all_routes.keys())
    machines_ids = set(pair[1] for pair in all_routes.keys())

    # Get Geodataframes for these specific nodes
    sources_gdf = graph_data.gdf_nodes[graph_data.gdf_nodes['node_id'].isin(sources_ids)]
    machines_gdf = graph_data.gdf_nodes[graph_data.gdf_nodes['node_id'].isin(machines_ids)]

    # Plot Sources (Green Triangles)
    if not sources_gdf.empty:
        sources_gdf.plot(ax=ax, color='lime', marker='^', markersize=100, 
                         edgecolor='black', label='Sources (Tenant)', zorder=4)

    # Plot Machines (Magenta Circles)
    if not machines_gdf.empty:
        machines_gdf.plot(ax=ax, color='magenta', marker='o', markersize=50,
                          edgecolor='black', label='Machines (Aboutissant)', zorder=4)

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper right', framealpha=0.9)
    # ---------------------------------------------------------

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    # Fixed to include extension so it saves successfully
    plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')


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

    edges_dict_raw = {tuple(sorted((int(r['i']), int(r['j'])))): r['length_m'] for _, r in graph_data1.gdf_segments.iterrows()}

    # Extract raw pairs from cables
    raw_pairs = [(int(r['tenant']), int(r['aboutissant'])) for _, r in graph_data1.df_cables.dropna().iterrows()]

    print(f"🔍 Original graph has {G_full.number_of_nodes()} nodes.")
    print(f"📉 Total connection pairs to process: {len(raw_pairs)}")

    # --- APPLY GRAPH REPAIR & ISOLATION ANALYSIS ---
    G_repaired, edges_dict, final_lcc = repair_graph_connectivity(G_full, edges_dict_raw, raw_pairs, graph_data1.gdf_nodes)

    # Filter pairs that are in the new, repaired LCC
    connection_pairs = []
    for src, tgt in raw_pairs:
        if src in final_lcc and tgt in final_lcc:
            connection_pairs.append((src, tgt))

    print(f"\n✅ Ready to route: {len(connection_pairs)}/{len(raw_pairs)} pairs reachable after repair.")

    # Filter angle map for valid nodes
    angle_map = {(int(r['i']), int(r['j']), int(r['k'])): r['abs_cosine']
                 for _, r in graph_data1.df_angles.iterrows()
                 if all(n in final_lcc for n in [int(r['i']), int(r['j']), int(r['k'])])}

    # --- DIJKSTRA INIT ---
    print("\n🧠 Computing Dijkstra Shortest Paths & Initializing Pheromones...")
    dijkstra_routes = {}
    global_init_pheromones = {e: 1.0 for e in edges_dict.keys()}

    for src, tgt in connection_pairs:
        try:
            d_path = nx.shortest_path(G_repaired, source=src, target=tgt, weight='weight')
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
    plot_network(graph_data1, dijkstra_routes, edges_dict_raw, title="Baseline: Dijkstra (Repaired Graph)", line_color='blue')

    # --- ACO PARAMETER TUNING EXECUTION ---
    print("\n🐜 Starting Ant Colony Optimization (ACO) Grid Search...")

    # 1. Define the parameter grid you want to test
    alphas = [0.5, 1.0, 1.5, 2.0]       # Pheromone importance
    betas = [0.5, 1.0, 1.5, 2.0]        # Distance importance
    gammas = [0.0, 0.5, 1.0]       # Angle (straightness) importance
    n_ants = 70
    n_iterations = 100

    # Generate all combinations of alpha, beta, and gamma
    param_grid = list(itertools.product(alphas, betas, gammas))
    print(f"Total parameter combinations to test: {len(param_grid)}")

    # List to store our metrics for the CSV
    results_log = []

    for a, b, g in param_grid:
        print(f"\n▶️ Running ACO with Alpha: {a}, Beta: {b}, Gamma: {g}...")

        params = {'alpha': a, 'beta': b, 'gamma': g, 'n_ants': n_ants, 'n_iterations': n_iterations}

        # Set up the worker
        worker_task = functools.partial(route_individual_pair, graph_gdf_segments=graph_data1.gdf_segments,
                                        angle_map=angle_map, edges_dict=edges_dict, params=params,
                                        initial_pheromones=global_init_pheromones)

        # Execute Multiprocessing
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(worker_task, connection_pairs))

        # Extract successful routes
        all_routes = {p: path for p, path, d in results if path}
        connected_count = len(all_routes)

        # Evaluate
        aco_metrics = evaluate_performance(all_routes, edges_dict, graph_data1)

        # Calculate savings compared to Dijkstra
        savings = 0
        if dijkstra_metrics.get("Trench_m", 0) > 0:
            savings = dijkstra_metrics["Trench_m"] - aco_metrics["Trench_m"]
            
        print(f"   Result: {connected_count}/{len(connection_pairs)} pairs connected.")
        print(f"   Trench: {aco_metrics['Trench_m']}m (Savings: {savings:.2f}m)")

        # Record data for CSV
        results_log.append({
            'Alpha': a,
            'Beta': b,
            'Gamma': g,
            'Connected_Pairs': connected_count,
            'Total_Cable_m': aco_metrics.get('Total_Cable_m', 0),
            'Total_Trench_m': aco_metrics.get('Trench_m', 0),
            'Trench_Savings_m': round(savings, 2)
        })

        # Dynamic Plot Title
        plot_title = f"ACO Network Layout (α={a}, β={b}, γ={g})"
        plot_network(graph_data1, all_routes, edges_dict_raw, title=plot_title, line_color='red')

    # --- SAVE RESULTS TO CSV ---
    df_results = pd.DataFrame(results_log)
    csv_filename = "aco_tuning_results.csv"
    df_results.to_csv(csv_filename, index=False)

    print(f"\n✅ Grid search complete! Results saved to '{csv_filename}'.")
    print("\nSummary of best runs:")
    # Sort by lowest trench distance and print top 3
    print(df_results.sort_values(by='Total_Trench_m').head(3))
