import numpy as np
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from IPython.display import clear_output  # Used for animating the plot in Jupyter Notebooks
import graph_data

# ==========================================
# 1. LOAD THE DATA
# ==========================================
file_path = "Capstone_M2DS_26/dataclass/graph_data_exemple.pkl"
# Make sure to replace this with your actual path if running outside your current directory
try:
    with open(file_path, "rb") as f:
        graph_data1 = pickle.load(f)
except FileNotFoundError:
    print(f"⚠️ Fichier introuvable : {file_path}. Assurez-vous que le chemin est correct.")

# ==========================================
# 2. ACO ROUTER CLASS
# ==========================================
class ACO_Router:
    def __init__(self, graph_data1, alpha=0.01, beta=1, rho=0.1, Q=50, gamma=2.0):
        self.graph = graph_data1
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.gamma = gamma

        self.edges = self._build_adjacency()
        self.pheromones = {edge: 1.0 for edge in self.edges.keys()}
        self.angle_map = self._build_angle_map()

        # Garde en mémoire les nœuds appartenant déjà au réseau de chaque source
        self.established_nodes = {}

    def run_global_routing(self, connection_pairs, n_ants=10, n_iterations=15, visualize_progress=True):
        global_network = {}
        failed_sources = set()

        G_check = nx.Graph()
        for edge, dist in self.edges.items():
            G_check.add_edge(edge[0], edge[1])

        for pair in connection_pairs:
            source = pair[0]
            target = pair[1]

            if pair in failed_sources:
                continue

            if source not in self.established_nodes:
                self.established_nodes[source] = {source}

            can_reach = any(nx.has_path(G_check, target, est_node)
                            for est_node in self.established_nodes[source])

            if not can_reach:
                print(f"❌ ABANDON : Source {source} inaccessible pour {target}. Ajout à la blacklist.")
                failed_sources.add(pair)
                continue

            print(f"Routage : Machine {target} -> Source {source} (Câblage dédié)...")
            
            # --- THE CRITICAL CHANGE ---
            # We no longer use self.established_nodes[source] as the target set.
            # We force the ant to route ALL the way back to the source node.
            target_set = {source} 

            best_path, best_dist = self.run_aco(target, target_set, n_ants, n_iterations, visualize=visualize_progress)

            if best_path:
                global_network[(source, target)] = best_path
                self._apply_heavy_reinforcement(best_path)
                # We still update established_nodes so your visualizer knows which trenches are active
                self.established_nodes[source].update(best_path)
            else:
                print(f"⚠️ ÉCHEC ACO : Pair {pair} abandonnée.")
                failed_sources.add(pair)

        return global_network

    def _apply_heavy_reinforcement(self, path):
        for i in range(len(path) - 1):
            edge = tuple(sorted((path[i], path[i+1])))
            self.pheromones[edge] += 1.0

    def _build_adjacency(self):
        adj = {}
        for _, row in self.graph.gdf_segments.iterrows():
            u, v, dist = int(row['i']), int(row['j']), row['length_m']
            edge = tuple(sorted((u, v)))
            adj[edge] = dist
        return adj

    def _get_neighbors(self, node):
        neighbors = []
        for edge, dist in self.edges.items():
            if node in edge:
                neighbor = edge[0] if edge[1] == node else edge[1]
                neighbors.append((neighbor, dist))
        return neighbors

    def _select_next_node(self, prev_node, current_node, gamma=1.5):
        neighbors = self._get_neighbors(current_node)

        # 1. Empêcher le demi-tour IMMÉDIAT (ping-pong) sauf si c'est une impasse
        candidates = [n for n in neighbors if n[0] != prev_node]
        if not candidates:
            # Impasse : on autorise la fourmi à revenir sur ses pas
            candidates = neighbors

        probabilities = []
        for neighbor, dist in candidates:
            edge = tuple(sorted((current_node, neighbor)))
            tau = self.pheromones[edge] ** self.alpha
            eta = (1.0 / dist) ** self.beta

            angle_factor = 1.0
            if prev_node is not None:
                cosine = self.angle_map.get((prev_node, current_node, neighbor), 0.5)
                angle_factor = (cosine + 0.1) ** gamma

            probabilities.append(tau * eta * angle_factor)

        sum_prob = sum(probabilities)
        if sum_prob == 0:
            return random.choice(candidates)[0]

        probabilities = [p / sum_prob for p in probabilities]
        return np.random.choice([c[0] for c in candidates], p=probabilities)

    def _build_angle_map(self):
        angle_map = {}
        for _, row in self.graph.df_angles.iterrows():
            angle_map[(int(row['i']), int(row['j']), int(row['k']))] = row['abs_cosine']
            angle_map[(int(row['k']), int(row['j']), int(row['i']))] = row['abs_cosine']
        return angle_map

    def run_aco(self, start_node, target_set, n_ants, n_iterations, visualize=False):
        best_path = None
        best_dist = float('inf')

        if start_node in target_set:
            return [start_node], 0.0

        for i in range(n_iterations):
            all_paths = []
            for _ in range(n_ants):
                path = self._construct_path(start_node, target_set)
                if path:
                    dist = self._calculate_path_length(path)
                    all_paths.append((path, dist))
                    if dist < best_dist:
                        best_dist = dist
                        best_path = path
            
            self._update_pheromones(all_paths)
            
            # --- PHEROMONE VISUALIZATION TRIGGER ---
            if visualize and (i%5 == 0):
                self._plot_pheromone_state(i + 1, start_node, target_set)

        return best_path, best_dist

    def _construct_path(self, start, target_set):
        path = [start]
        prev = None
        current = start

        # Limite élargie car la fourmi peut faire des détours avant d'effacer ses boucles
        max_steps = len(self.edges) * 3
        for _ in range(max_steps):
            next_node = self._select_next_node(prev, current)

            if next_node is None:
                return None

            # 2. LOOP ERASURE : Si la fourmi croise son propre chemin, on coupe la boucle
            if next_node in path:
                loop_start_index = path.index(next_node)
                path = path[:loop_start_index + 1]  # On garde le chemin jusqu'au point d'intersection
                current = next_node
                prev = path[-2] if len(path) > 1 else None
            else:
                path.append(next_node)
                if next_node in target_set:
                    return path  # Cible atteinte
                prev = current
                current = next_node
        return None

    def _calculate_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            edge = tuple(sorted((path[i], path[i+1])))
            length += self.edges[edge]
        return length

    def _update_pheromones(self, all_paths):
        for edge in self.pheromones:
            self.pheromones[edge] *= (1 - self.rho)
        for path, dist in all_paths:
            if dist > 0:
                for i in range(len(path) - 1):
                    edge = tuple(sorted((path[i], path[i+1])))
                    self.pheromones[edge] += (self.Q / dist)

    def _plot_pheromone_state(self, iteration, start_node, target_set):
        """Generates a visualization showing global pheromone memory but only one Source and one Machine."""
        # 1. Update pheromone data in the GeoDataFrame
        phero_levels = []
        for _, row in self.graph.gdf_segments.iterrows():
            edge = tuple(sorted((int(row['i']), int(row['j']))))
            phero_levels.append(self.pheromones.get(edge, 1.0))
        
        self.graph.gdf_segments['current_pheromone'] = phero_levels

        # 2. Setup the plot
        clear_output(wait=True) 
        fig, ax = plt.subplots(figsize=(14, 9))

        # --- LAYER 1: THE INFRASTRUCTURE SKELETON ---
        self.graph.gdf_segments.plot(ax=ax, color='gray', linewidth=0.7, alpha=0.3, zorder=1)

        # --- LAYER 2: GLOBAL MEMORY (Previous established routes) ---
        # Show paths established by previous machine routings in dark maroon
        global_memory_mask = self.graph.gdf_segments['current_pheromone'] > 100
        if global_memory_mask.any():
            global_edges = self.graph.gdf_segments[global_memory_mask]
            global_edges.plot(ax=ax, color='#660033', linewidth=3, alpha=0.3, label='Mémoire des trajets précédents')

        # --- LAYER 3: CURRENT EXPLORATION (Active ants) ---
        # Show the current ants' work in bright red
        active_mask = (self.graph.gdf_segments['current_pheromone'] > 1.1) & (~global_memory_mask)
        if active_mask.any():
            active_edges = self.graph.gdf_segments[active_mask].copy()
            max_p = active_edges['current_pheromone'].max()
            active_edges['linewidth'] = 1.0 + (active_edges['current_pheromone'] / max_p) * 3
            active_edges.plot(ax=ax, color='red', linewidth=active_edges['linewidth'], alpha=0.8, label='Exploration Fourmis')

        # --- LAYER 4: SPECIFIC SOURCE AND MACHINE ONLY ---
        gdf_nodes_idx = self.graph.gdf_nodes.set_index('node_id')
        
        # We only want to plot the ORIGINAL Source (the very first node in the target set)
        # Assuming the first node in target_set is the main Source/Hub
        source_id = list(target_set)[0] 
        
        if source_id in gdf_nodes_idx.index:
            src = gdf_nodes_idx.loc[source_id, 'geometry']
            ax.scatter([src.x], [src.y], color='#00ff00', s=150, marker='^', 
                       edgecolors='black', label=f'Source Unique ({source_id})', zorder=6)

        # The specific machine we are currently routing
        if start_node in gdf_nodes_idx.index:
            sn = gdf_nodes_idx.loc[start_node, 'geometry']
            ax.scatter([sn.x], [sn.y], color='blue', s=120, marker='o', 
                       edgecolors='white', label=f'Machine Actuelle ({start_node})', zorder=7)

        plt.title(f"ACO iteration {iteration} | Connexion: Machine {start_node} -> Source {source_id}", fontsize=14)
        plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)
        plt.savefig(f"Progression ACO - Itération {iteration} | Recherche depuis la Machine {start_node}", dpi=300, bbox_inches='tight')
        plt.close(fig)


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def evaluate_routing_performance(all_routes, router):
    total_dedicated_cable_length = 0
    unique_edges = set()

    for path in all_routes.values():
        # 1. Add up the total length of every individual cable (Copper Wire)
        total_dedicated_cable_length += router._calculate_path_length(path)
        
        # 2. Collect unique edges to measure the physical trench footprint (Infrastructure)
        for i in range(len(path) - 1):
            unique_edges.add(tuple(sorted((path[i], path[i+1]))))

    # Calculate the total length of just the shared infrastructure
    total_infrastructure_trench_length = sum(router.edges[e] for e in unique_edges)

    # Calculate Sinuosity (Straightness)
    sinuosity_scores = []
    for (src, tgt), path in all_routes.items():
        if len(path) < 2:
            continue

        actual_dist = router._calculate_path_length(path)
        node_start = router.graph.gdf_nodes.set_index('node_id').loc[path[0], 'geometry']
        node_end = router.graph.gdf_nodes.set_index('node_id').loc[path[-1], 'geometry']
        direct_dist = node_start.distance(node_end)

        if direct_dist > 0:
            sinuosity_scores.append(actual_dist / direct_dist)

    avg_sinuosity = np.mean(sinuosity_scores) if sinuosity_scores else 0

    # Calculate Bending Cost (Turns)
    total_bending_cost = 0
    for path in all_routes.values():
        for i in range(len(path) - 2):
            cosine = router.angle_map.get((path[i], path[i+1], path[i+2]), 1.0)
            total_bending_cost += (1.0 - cosine)

    return {
        "Total_Dedicated_Cable_Length": round(total_dedicated_cable_length, 2),
        "Total_Infrastructure_Trench_Length": round(total_infrastructure_trench_length, 2),
        "Avg_Path_Sinuosity": round(avg_sinuosity, 3),
        "Total_Bending_Cost": round(total_bending_cost, 2)
    }


def plot_global_network(graph_data1, all_routes, connection_pairs):
    fig, ax = plt.subplots(figsize=(15, 10))

    graph_data1.gdf_segments.plot(ax=ax, color='black', linewidth=0.2, alpha=0.3, label='Segments possibles')

    for (src, tgt), path in all_routes.items():
        if not path or len(path) < 2:
            continue

        path_edges = [tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)]
        mask = graph_data1.gdf_segments.apply(
            lambda row: tuple(sorted((int(row['i']), int(row['j'])))) in path_edges, axis=1
        )
        graph_data1.gdf_segments[mask].plot(ax=ax, color='red', linewidth=1.2, alpha=0.7)

    sources_ids = list(set([p[0] for p in connection_pairs]))
    targets_ids = list(set([p[1] for p in connection_pairs]))

    gdf_nodes = graph_data1.gdf_nodes
    sources_points = gdf_nodes[gdf_nodes['node_id'].astype(int).isin(sources_ids)]
    targets_points = gdf_nodes[gdf_nodes['node_id'].astype(int).isin(targets_ids)]

    sources_points.plot(ax=ax, color='green', markersize=50, marker='^', label='Sources (Tenants)', zorder=5)
    targets_points.plot(ax=ax, color='blue', markersize=30, marker='o', label='Machines (Aboutissants)', zorder=5)

    plt.title("Optimisation ACO : Réseau final de cuivre", fontsize=14)
    plt.legend(loc='upper right')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.savefig("network_sources_machines.png", dpi=300, bbox_inches='tight')
    plt.show()

def run_aco_grid_search(graph_data1, subset_pairs, param_grid):
    results = []
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Starting Grid Search: {len(param_combinations)} total combinations to test.\n")

    for i, params in enumerate(param_combinations):
        print(f"--- Run {i+1}/{len(param_combinations)} | Testing Params: {params} ---")

        alpha = params.get('alpha', 3.0)
        beta = params.get('beta', 3.0)
        gamma = params.get('gamma', 3.5)
        n_ants = 20
        n_iterations = 15

        router = ACO_Router(graph_data1, alpha=alpha, beta=beta, gamma=gamma)
        
        # NOTE: visualize_progress is left as False here to prevent crashing during grid search
        all_routes = router.run_global_routing(subset_pairs, n_ants=n_ants, n_iterations=n_iterations, visualize_progress=True)

        metrics = evaluate_routing_performance(all_routes, router)
        success_rate = len(all_routes) / len(subset_pairs) if len(subset_pairs) > 0 else 0

        run_result = {**params, **metrics, "Success_Rate": round(success_rate, 2)}
        results.append(run_result)

    df_results = pd.DataFrame(results)
    return df_results


# ==========================================
# 4. DATA PREP & EXECUTION
# ==========================================
if 'graph_data1' in locals():
    # 1. Connected components check
    G = nx.Graph()
    for _, row in graph_data1.gdf_segments.iterrows():
        G.add_edge(int(row['i']), int(row['j']), weight=row['length_m'])

    largest_cc = max(nx.connected_components(G), key=len)
    valid_node_ids = set(largest_cc)
    print(f"Plus grand composant connexe identifié : {len(valid_node_ids)} nœuds utiles.\n")

    # 2. Clean and Filter
    df_cables_clean = graph_data1.df_cables.dropna(subset=['tenant', 'aboutissant'])
    df_final = df_cables_clean[
        df_cables_clean['tenant'].astype(int).isin(valid_node_ids) &
        df_cables_clean['aboutissant'].astype(int).isin(valid_node_ids)
    ].copy()

    connection_pairs = []
    for _, row in df_final.iterrows():
        connection_pairs.append((int(row['tenant']), int(row['aboutissant'])))

    print(f"✅ Prêt pour le routage de {len(connection_pairs)} connexions valides.\n")

    if connection_pairs:
        """
        # ---------------------------------------------------------
        # DEMO : PHEROMONE VISUALIZATION ON A SINGLE PAIR
        # ---------------------------------------------------------
        print("\n" + "="*50)
        print("🔍 DÉMO VISUELLE : Routage d'une seule paire...")
        print("="*50)
        
        test_pair = [connection_pairs[0]] 
        demo_router = ACO_Router(graph_data1, alpha=1.5, beta=2.0, gamma=1.5)
        
        # Turn visualize_progress=True to see the map update!
        demo_routes = demo_router.run_global_routing(
            test_pair, 
            n_ants=15, 
            n_iterations=10, 
            visualize_progress=True 
        )
        print("✅ Démo visuelle terminée.\n")
        """
        # ---------------------------------------------------------
        # FULL RUN : GRID SEARCH (Visualization OFF)
        # ---------------------------------------------------------
        print("\n" + "="*50)
        print("🚀 DÉMARRAGE DU GRID SEARCH SUR TOUTES LES PAIRES...")
        print("="*50)

        param_grid = {
            'alpha': [0.01],
            'beta': [1.0],
            'gamma': [2],
        }

        df_grid_results = run_aco_grid_search(graph_data1, connection_pairs, param_grid)

        print("\n🏆 GRID SEARCH TERMINÉ ! Top 5 configurations :")
        df_sorted = df_grid_results.sort_values(by=['Total_Copper_Length', 'Success_Rate'], ascending=[True, False])
        print(df_sorted.head().to_string(index=False))

        df_sorted.to_csv("aco_grid_search_results.csv", index=False)

        best_params = df_sorted.iloc[0]
        print(f"\n🗺️ Génération du tracé final pour la meilleure configuration : {best_params.to_dict()}")

        best_router = ACO_Router(
            graph_data1,
            alpha=best_params['alpha'],
            beta=best_params['beta'],
            gamma=best_params['gamma']
        )

        best_routes = best_router.run_global_routing(
            connection_pairs,
            n_ants=10,
            n_iterations=16,
            visualize_progress=True # Keep false for the final full run!
        )

        plot_global_network(graph_data1, best_routes, connection_pairs)
