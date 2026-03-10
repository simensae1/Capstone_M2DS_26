import numpy as np
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
import geopandas as gpd
import graph_data
import networkx as nx
from shapely.geometry import Point
from shapely.ops import nearest_points
import itertools

# 1. Load the data
file_path = "Capstone_M2DS_26/dataclass/graph_data_exemple.pkl"
with open(file_path, "rb") as f:
    graph_data1 = pickle.load(f)


class ACO_Router:
    def __init__(self, graph_data1, alpha=1.0, beta=2.0, rho=0.1, Q=100, gamma=1.5):
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

    def run_global_routing(self, connection_pairs, n_ants=10, n_iterations=15):
        global_network = {}
        failed_sources = set()  # Pour stocker les sources qui ne fonctionnent pas

        G_check = nx.Graph()
        for edge, dist in self.edges.items():
            G_check.add_edge(edge[0], edge[1])

        for pair in connection_pairs:
            source = pair[0]
            target = pair[1]

            # Si la source est dans la liste noire, on passe directement à la suite
            if pair in failed_sources:
                continue

            if source not in self.established_nodes:
                self.established_nodes[source] = {source}

            # Vérification de connectivité théorique
            can_reach = any(nx.has_path(G_check, target, est_node) 
                            for est_node in self.established_nodes[source])

            if not can_reach:
                print(f"❌ ABANDON : Source {source} inaccessible pour {target}. Ajout à la blacklist.")
                failed_sources.add(pair)  # On blacklist la source
                continue

            print(f"Routage : Machine {target} -> Réseau de la Source {source}...")
            target_set = self.established_nodes[source]

            best_path, best_dist = self.run_aco(target, target_set, n_ants, n_iterations)

            if best_path:
                global_network[(source, target)] = best_path
                self._apply_heavy_reinforcement(best_path)
                self.established_nodes[source].update(best_path)
            else:
                print(f"⚠️ ÉCHEC ACO : Pair {pair} abandonnée.")
                failed_sources.add(pair)  # On blacklist la source ici aussi

        return global_network

    def _apply_heavy_reinforcement(self, path):
        for i in range(len(path) - 1):
            edge = tuple(sorted((path[i], path[i+1])))
            self.pheromones[edge] += 500.0

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

    def run_aco(self, start_node, target_set, n_ants, n_iterations):
        best_path = None
        best_dist = float('inf')

        if start_node in target_set:
            return [start_node], 0.0

        for _ in range(n_iterations):
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
        return best_path, best_dist

    def _construct_path(self, start, target_set):
        path = [start]
        prev = None
        current = start

        # Limite élargie car la fourmi peut faire des détours avant d'effacer ses boucles
        max_steps = len(self.edges) * 10 

        for _ in range(max_steps):
            # La notion de "visited" est supprimée ici pour permettre l'exploration totale
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


# --- EXECUTION MODIFIÉE AVEC VÉRIFICATION DES NŒUDS ET COMPOSANTS CONNEXES ---

# 1. Créer un graphe NetworkX temporaire pour isoler le composant connexe principal
G = nx.Graph()
for _, row in graph_data1.gdf_segments.iterrows():
    G.add_edge(int(row['i']), int(row['j']), weight=row['length_m'])

# Extraire le plus grand composant connexe (pour s'assurer que les chemins sont possibles)
largest_cc = max(nx.connected_components(G), key=len)
valid_node_ids = set(largest_cc)
print(f"Plus grand composant connexe identifié : {len(valid_node_ids)} nœuds utiles sur le réseau total.")

# 2. Nettoyage initial (suppression des NaN)
df_cables_clean = graph_data1.df_cables.dropna(subset=['tenant', 'aboutissant'])

# 3. Filtrage : On ne garde que les câbles dont les DEUX extrémités sont dans le composant connexe
initial_count = len(df_cables_clean)
df_final = df_cables_clean[
    df_cables_clean['tenant'].astype(int).isin(valid_node_ids) &
    df_cables_clean['aboutissant'].astype(int).isin(valid_node_ids)
].copy()

dropped_count = initial_count - len(df_final)
if dropped_count > 0:
    print(f"⚠️ Attention : {dropped_count} lignes ignorées (nœuds hors du graphe principal/composant connexe).")

# 4. Créer la liste des paires (Source, Machine)
connection_pairs = []
for _, row in df_final.iterrows():
    connection_pairs.append((int(row['tenant']), int(row['aboutissant'])))

print(f"✅ Prêt pour le routage de {len(connection_pairs)} connexions valides.")
"""
# 5. Lancer le routage global
router = ACO_Router(graph_data1, alpha=1.5, beta=1.0, gamma=2.0)
all_routes = router.run_global_routing(connection_pairs)

# 6. Calculer la longueur totale du cuivre (les arêtes mutualisées ne sont comptées qu'une fois)
unique_edges = set()
for path in all_routes.values():
    for i in range(len(path)-1):
        unique_edges.add(tuple(sorted((path[i], path[i+1]))))

total_dist = sum(router.edges[e] for e in unique_edges)
print(f"\n📏 Longueur totale du câble utilisé (mutualisé) : {total_dist:.2f} m")
"""


def evaluate_routing_performance(all_routes, router):
    """
    Calcule les métriques de performance pour évaluer la qualité du routage.
    """
    # 1. COÛT DE L'INFRASTRUCTURE (Longueur totale du cuivre mutualisé)
    unique_edges = set()
    for path in all_routes.values():
        for i in range(len(path) - 1):
            unique_edges.add(tuple(sorted((path[i], path[i+1]))))
    
    total_infrastructure_length = sum(router.edges[e] for e in unique_edges)

    # 2. COÛT DE SINUOSITÉ (Efficacité individuelle des chemins)
    # Ratio : Longueur parcourue / Distance à vol d'oiseau (ou Dijkstra si disponible)
    sinuosity_scores = []
    for (src, tgt), path in all_routes.items():
        if len(path) < 2: continue
        
        actual_dist = router._calculate_path_length(path)
        # Approximation simple par distance euclidienne entre points de départ/arrivée
        node_start = router.graph.gdf_nodes.set_index('node_id').loc[path[0], 'geometry']
        node_end = router.graph.gdf_nodes.set_index('node_id').loc[path[-1], 'geometry']
        direct_dist = node_start.distance(node_end)
        
        if direct_dist > 0:
            sinuosity_scores.append(actual_dist / direct_dist)

    avg_sinuosity = np.mean(sinuosity_scores) if sinuosity_scores else 0

    # 3. COÛT DE COURBURE (Difficulté d'installation)
    # Somme des angles pour tout le réseau
    total_bending_cost = 0
    for path in all_routes.values():
        for i in range(len(path) - 2):
            # On utilise le complémentaire du cosinus : 0 si c'est droit (cos=1), 1 si c'est à 90°
            cosine = router.angle_map.get((path[i], path[i+1], path[i+2]), 1.0)
            total_bending_cost += (1.0 - cosine)

    return {
        "Total_Copper_Length": round(total_infrastructure_length, 2),
        "Avg_Path_Sinuosity": round(avg_sinuosity, 3),
        "Total_Bending_Cost": round(total_bending_cost, 2)
    }

# 7. Affichage


def plot_global_network(graph_data1, all_routes, connection_pairs):
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 1. Dessiner le fond de carte (tous les segments possibles)
    graph_data1.gdf_segments.plot(ax=ax, color='black', linewidth=0.2, alpha=0.3, label='Segments possibles')

    # 2. Dessiner les routes trouvées par les fourmis
    for (src, tgt), path in all_routes.items():
        if not path or len(path) < 2:
            continue
            
        path_edges = [tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)]
        mask = graph_data1.gdf_segments.apply(
            lambda row: tuple(sorted((int(row['i']), int(row['j'])))) in path_edges, axis=1
        )
        graph_data1.gdf_segments[mask].plot(ax=ax, color='red', linewidth=1.2, alpha=0.7)

    # 3. Extraire et afficher les Sources (Tenants) et Machines (Aboutissants)
    # On récupère les IDs uniques pour ne pas surcharger le plot
    sources_ids = list(set([p[0] for p in connection_pairs]))
    targets_ids = list(set([p[1] for p in connection_pairs]))

    gdf_nodes = graph_data1.gdf_nodes
    # Filtrer les points géométriques
    sources_points = gdf_nodes[gdf_nodes['node_id'].astype(int).isin(sources_ids)]
    targets_points = gdf_nodes[gdf_nodes['node_id'].astype(int).isin(targets_ids)]

    # Affichage des points
    sources_points.plot(ax=ax, color='green', markersize=50, marker='^', label='Sources (Tenants)', zorder=5)
    targets_points.plot(ax=ax, color='blue', markersize=30, marker='o', label='Machines (Aboutissants)', zorder=5)

    plt.title("Optimisation ACO : Réseau de cuivre et points de connexion", fontsize=14)
    plt.legend(loc='upper right')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    plt.savefig("network_sources_machines.png", dpi=300, bbox_inches='tight')
    plt.show()

"""
# Appel de la fonction mis à jour
plot_global_network(graph_data1, all_routes, connection_pairs)
"""


def run_aco_grid_search(graph_data1, subset_pairs, param_grid):
    """
    Executes a grid search over specified ACO parameters and returns a DataFrame of results.
    """
    results = []

    # Extract keys and generate all combinations of the parameter lists
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Starting Grid Search: {len(param_combinations)} total combinations to test.\n")

    for i, params in enumerate(param_combinations):
        print(f"--- Run {i+1}/{len(param_combinations)} | Testing Params: {params} ---")

        # Isolate initialization params vs runtime params
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 2.0)
        gamma = params.get('gamma', 1.5)
        n_ants = 3
        n_iterations = 5

        # 1. Initialize the Router with current grid parameters
        router = ACO_Router(graph_data1, alpha=alpha, beta=beta, gamma=gamma)

        # 2. Run the routing on the subset of pairs
        all_routes = router.run_global_routing(subset_pairs, n_ants=n_ants, n_iterations=n_iterations)

        # 3. Evaluate performance
        metrics = evaluate_routing_performance(all_routes, router)

        # 4. Calculate success rate (how many pairs were successfully routed)
        success_rate = len(all_routes) / len(subset_pairs) if len(subset_pairs) > 0 else 0

        # 5. Store the results
        run_result = {**params, **metrics, "Success_Rate": round(success_rate, 2)}
        results.append(run_result)

    # Convert to a DataFrame for easy sorting and visualization
    df_results = pd.DataFrame(results)
    return df_results

# Define the parameter grid
# Be careful: adding too many values will exponentially increase runtime!


param_grid = {
    'alpha': [1.0, 1.5],          # Pheromone importance
    'beta': [1.0, 2.0],           # Distance importance
    'gamma': [1.0, 2.0],          # Angle/Straightness importance
}

# Run the Grid Search
df_grid_results = run_aco_grid_search(graph_data1, connection_pairs, param_grid)

# --- ANALYZE RESULTS ---

print("\n GRID SEARCH TERMINE ! Voici les 5 meilleures configurations (triées par coût d'infrastructure le plus bas) :")
# Sort by the lowest total copper length, then by highest success rate
df_sorted = df_grid_results.sort_values(by=['Total_Copper_Length', 'Success_Rate'], ascending=[True, False])
print(df_sorted.head().to_string(index=False))

# Optional: Save results to CSV for external analysis
df_sorted.to_csv("aco_grid_search_results.csv", index=False)

# --- VISUALIZE THE BEST RUN ---
# Automatically extract the best parameters and plot that specific network
best_params = df_sorted.iloc[0]
print(f"\n Génération du tracé pour la meilleure configuration : {best_params.to_dict()}")

best_router = ACO_Router(
    graph_data1,
    alpha=best_params['alpha'],
    beta=best_params['beta'],
    gamma=best_params['gamma']
)

best_routes = best_router.run_global_routing(
    connection_pairs,
    n_ants=3,
    n_iterations=5
)

plot_global_network(graph_data1, best_routes, connection_pairs)
