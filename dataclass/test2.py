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

    def run_global_routing(self, connection_pairs, n_ants=20, n_iterations=30):
        """
        Relie chaque machine à SA source spécifique.
        connection_pairs: liste de tuples (source, cible)
        """
        global_network = {}

        # On parcourt chaque paire (Source spécifique -> Machine spécifique)
        for source, target in connection_pairs:
            print(f"Routage : Source {source} -> Machine {target}...")

            # Les phéromones accumulées aident à mutualiser les segments si les chemins se croisent
            best_path, _ = self.run_aco(source, target, n_ants, n_iterations)

            if best_path:
                # On utilise la cible comme clé (ou le tuple (source, cible) si une machine a plusieurs sources)
                global_network[(source, target)] = best_path
                self._apply_heavy_reinforcement(best_path)
            else:
                print(f"⚠️ Impossible de relier la source {source} à la machine {target}")

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

    def _select_next_node(self, prev_node, current_node, visited, gamma=1.5):
        neighbors = self._get_neighbors(current_node)
        candidates = [n for n in neighbors if n[0] not in visited]

        if not candidates:
            return None

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

    def run_aco(self, start_node, end_node, n_ants=10, n_iterations=50):
        best_path = None
        best_dist = float('inf')
        for _ in range(n_iterations):
            all_paths = []
            for _ in range(n_ants):
                path = self._construct_path(start_node, end_node)
                if path:
                    dist = self._calculate_path_length(path)
                    all_paths.append((path, dist))
                    if dist < best_dist:
                        best_dist = dist
                        best_path = path
            self._update_pheromones(all_paths)
        return best_path, best_dist

    def _construct_path(self, start, end):
        path = [start]
        visited = {start}
        prev = None
        current = start

        for _ in range(len(self.edges) * 5):  # Limite de recherche
            if current == end:
                return path
            next_node = self._select_next_node(prev, current, visited)
            if next_node is None:
                return None
            path.append(next_node)
            visited.add(next_node)
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
            for i in range(len(path) - 1):
                edge = tuple(sorted((path[i], path[i+1])))
                self.pheromones[edge] += (self.Q / dist)

# --- EXECUTION MODIFIÉE AVEC VÉRIFICATION DES NŒUDS ---
# 1. Identifier les nœuds réellement présents dans le graphe spatial


valid_node_ids = set(graph_data1.gdf_nodes['node_id'].astype(int))

# 2. Nettoyage initial (suppression des NaN)
df_cables_clean = graph_data1.df_cables.dropna(subset=['tenant', 'aboutissant'])

# 3. Filtrage : On ne garde que les câbles dont les deux extrémités existent dans le graphe
initial_count = len(df_cables_clean)
df_final = df_cables_clean[
    df_cables_clean['tenant'].astype(int).isin(valid_node_ids) &
    df_cables_clean['aboutissant'].astype(int).isin(valid_node_ids)
].copy()

dropped_count = initial_count - len(df_final)
if dropped_count > 0:
    print(f"⚠️ Attention : {dropped_count} lignes ignorées car les IDs de nœuds n'existent pas dans gdf_nodes.")

# 4. Créer la liste des paires (Source, Machine)
connection_pairs = []
for _, row in df_final.iterrows():
    connection_pairs.append((int(row['tenant']), int(row['aboutissant'])))

print(f"✅ Prêt pour le routage de {len(connection_pairs)} connexions valides.")

# 5. Lancer le routage global
router = ACO_Router(graph_data1, alpha=1.5, beta=1.0, gamma=2.0)
all_routes = router.run_global_routing(connection_pairs)

# 6. Calculer la longueur totale du cuivre
unique_edges = set()
for path in all_routes.values():
    for i in range(len(path)-1):
        unique_edges.add(tuple(sorted((path[i], path[i+1]))))

total_dist = sum(router.edges[e] for e in unique_edges)
print(f"\n📏 Longueur totale du cuivre utilisé : {total_dist:.2f} m")

# 4. Affichage


def plot_global_network(graph_data1, all_routes):
    fig, ax = plt.subplots(figsize=(15, 10))
    graph_data1.gdf_segments.plot(ax=ax, color='black', linewidth=0.2, alpha=0.3)

    for (src, tgt), path in all_routes.items():
        path_edges = [tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)]
        mask = graph_data1.gdf_segments.apply(
            lambda row: tuple(sorted((int(row['i']), int(row['j'])))) in path_edges, axis=1
        )
        graph_data1.gdf_segments[mask].plot(ax=ax, color='red', linewidth=1.5, alpha=0.6)

    plt.title("Réseau Multi-Sources Optimisé")
    plt.savefig("best_path_ant_multple_sources.png", dpi=300)
    plt.show()


plot_global_network(graph_data1, all_routes)
