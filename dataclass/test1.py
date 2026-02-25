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

print(graph_data1)


def snap_machines_to_network(graph_data1):
    """
    Connecte chaque nœud machine au segment le plus proche 
    s'il n'est pas déjà connecté.
    """
    for _, machine in graph_data1.gdf_nodes.iterrows():
        node_id = machine['node_id']
        node_point = machine.geometry

        # Vérifier si ce nœud a déjà des segments connectés
        is_connected = any((graph_data1.gdf_segments['i'] == node_id) |
                           (graph_data1.gdf_segments['j'] == node_id))

        if not is_connected:
            # Trouver le segment le plus proche
            nearest_seg = graph_data1.gdf_segments.geometry.distance(node_point).idxmin()
            segment = graph_data1.gdf_segments.loc[nearest_seg]

            print(f"Connexion forcée : Machine {node_id} attachée au segment {nearest_seg}")

            # Ajouter cette connexion dans gdf_segments
            # Note: Ici, il faudrait ajouter une nouvelle ligne à gdf_segments
            # reliant le node_id au nœud le plus proche sur le segment.


class ACO_Router:
    def __init__(self, graph_data1, alpha=1.0, beta=2.0, rho=0.1, Q=100, gamma=1.5):
        self.graph = graph_data1
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.gamma = gamma  # Importance de l'angle

        # Initialisation des structures de données
        self.edges = self._build_adjacency()
        self.pheromones = {edge: 1.0 for edge in self.edges.keys()}

        # --- LA LIGNE MANQUANTE ÉTAIT ICI ---
        self.angle_map = self._build_angle_map()

    def run_global_routing(self, source_node, target_nodes, n_ants=20, n_iterations=30):
            """
            Relie toutes les cibles à la source en encourageant le partage de segments.
            """
            global_network = {}  # Pour stocker les chemins finaux de chaque machine

            # On trie les cibles par distance à la source (optionnel mais efficace)
            # On traite machine par machine
            for target in target_nodes:
                print(f"Routage vers la machine {target}...")

                # On fait courir les fourmis pour cette cible spécifique
                # Note : les phéromones des machines précédentes RESTENT sur le graphe !
                best_path, _ = self.run_aco(source_node, target, n_ants, n_iterations)

                if best_path:
                    global_network[target] = best_path
                    # RÉCOMPENSE MASSIVE : on booste les phéromones du chemin trouvé
                    # pour que les machines suivantes "aient envie" de l'emprunter.
                    self._apply_heavy_reinforcement(best_path)
                else:
                    print(f"⚠️ Impossible de relier la machine {target}")

            return global_network

    def _apply_heavy_reinforcement(self, path):
        """Dépose une phéromone 'permanente' pour encourager la mutualisation."""
        for i in range(len(path) - 1):
            edge = tuple(sorted((path[i], path[i+1])))
            self.pheromones[edge] += 500.0  # Valeur élevée pour créer un "tronc commun"

    def _build_adjacency(self):
        """Transforme gdf_segments en dictionnaire d'adjacence { (i, j): length }"""
        adj = {}
        for _, row in self.graph.gdf_segments.iterrows():
            u, v, dist = int(row['i']), int(row['j']), row['length_m']
            # On stocke l'arête de manière bidirectionnelle (triée pour la clé)
            edge = tuple(sorted((u, v)))
            adj[edge] = dist
        return adj

    def _get_neighbors(self, node):
        """Retourne les voisins d'un nœud et la distance associée"""
        neighbors = []
        for edge, dist in self.edges.items():
            if node in edge:
                neighbor = edge[0] if edge[1] == node else edge[1]
                neighbors.append((neighbor, dist))
        return neighbors

    def _select_next_node(self, prev_node, current_node, visited, gamma=1.5):
            neighbors = self._get_neighbors(current_node)
            candidates = neighbors

            if not candidates:
                return None

            probabilities = []
            for neighbor, dist in candidates:
                edge = tuple(sorted((current_node, neighbor)))

                # 1. Phéromone (Expérience collective)
                tau = self.pheromones[edge] ** self.alpha

                # 2. Visibilité (Distance locale)
                eta = (1.0 / dist) ** self.beta

                # 3. Facteur d'angle (Fluidité du tracé)
                # Si c'est le premier nœud, pas de direction précédente, angle neutre (1.0)
                angle_factor = 1.0
                if prev_node is not None:
                    # On cherche l'angle formé par le triplet (précédent, actuel, suivant)
                    cosine = self.angle_map.get((prev_node, current_node, neighbor), 0.5)
                    # On utilise le cosinus : plus il est proche de 1 (droit), plus le score est haut
                    angle_factor = (cosine + 0.1) ** gamma

                probabilities.append(tau * eta * angle_factor)

            sum_prob = sum(probabilities)
            if sum_prob == 0:
                return random.choice(candidates)[0]

            probabilities = [p / sum_prob for p in probabilities]
            return np.random.choice([c[0] for c in candidates], p=probabilities)

    def _build_angle_map(self):
        """Crée un dictionnaire {(i, j, k): abs_cosine} pour un accès rapide"""
        angle_map = {}
        for _, row in self.graph.df_angles.iterrows():
            # On stocke dans les deux sens de parcours du triplet
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
            current = start
            while current != end:
                next_node = self._select_next_node(None, current, visited) # prev_node ignoré ici pour le test
                if next_node is None:
                    print(f"DEBUG: Fourmi bloquée au nœud {current} (Target: {end})")
                    return None
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            return path

    def _calculate_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            edge = tuple(sorted((path[i], path[i+1])))
            length += self.edges[edge]
        return length

    def _update_pheromones(self, all_paths):
        # Évaporation
        for edge in self.pheromones:
            self.pheromones[edge] *= (1 - self.rho)

        # Dépôt
        for path, dist in all_paths:
            for i in range(len(path) - 1):
                edge = tuple(sorted((path[i], path[i+1])))
                self.pheromones[edge] += (self.Q / dist)


# 1. Identifier la source (on prend le premier 'tenant' comme référence)
source = int(graph_data1.df_cables.iloc[0]['tenant'])

# 2. Identifier toutes les machines uniques à relier
machines = graph_data1.df_cables['aboutissant'].unique().astype(int).tolist()

# 3. Lancer le routage global
router = ACO_Router(graph_data1, alpha=1.5, beta=1.0, gamma=2.0)  # On augmente alpha pour suivre les pistes
all_routes = router.run_global_routing(source, machines)
print(f"Voisins du nœud 64 : {router._get_neighbors(64)}")

# 4. Calculer la longueur totale du réseau (en comptant chaque segment unique une seule fois)
unique_edges = set()
for path in all_routes.values():
    for i in range(len(path)-1):
        unique_edges.add(tuple(sorted((path[i], path[i+1]))))

total_dist = sum(router.edges[e] for e in unique_edges)
print(f"Longueur totale du cuivre utilisé : {total_dist:.2f} m")


def plot_global_network(graph_data1, all_routes):
    fig, ax = plt.subplots(figsize=(15, 10))
    graph_data1.gdf_segments.plot(ax=ax, color='black', linewidth=0.2, alpha=0.3)

    # Palette de couleurs pour distinguer les chemins si besoin
    # Mais ici on veut voir le réseau global
    for target, path in all_routes.items():
        path_edges = [tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)]
        mask = graph_data1.gdf_segments.apply(
            lambda row: tuple(sorted((int(row['i']), int(row['j'])))) in path_edges, axis=1
        )
        graph_data1.gdf_segments[mask].plot(ax=ax, color='red', linewidth=1.5, alpha=0.6)

    plt.title("Réseau de Câblage Optimisé (Mutualisation des segments)")
    plt.savefig("best_path_ant.png", dpi=300)
    plt.show()


plot_global_network(graph_data1, all_routes)
