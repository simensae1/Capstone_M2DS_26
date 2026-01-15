# -*- coding: utf-8 -*-
############
# Packages #
############
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import igraph as ig
import networkx as nx
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))
#######################
# Internal Imports #
#######################
from dataclass.graph_data import GraphData
from solver.base_solver import BaseSolverInterface
from solver.hgraph.base_hgraph import BaseHGraph
from solver.hgraph.hgraph_simple import GraphSimple
from solver.utils.graph_builders import (
    build_igraph_graph,
    build_networkx_graph,
    build_networkx_segment_graph,
)
from solver.utils.compute_metrics import shortest_path_running_metrics
from solver.utils.cost_function import compute_objective_from_simplehgraph

#########
# Utils #
#########

import logging
logger = logging.getLogger(__name__)


class ShortestPathSolver(BaseSolverInterface):
    """Solveur de plus court chemin."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        config = config or {}
        model_id = config.get("model_id", "shortest_path_solver")
        super().__init__(
            model_id=model_id,
            config=config,
        )
        graph_type = self.config.get("graph_type", "networkx").lower()
        if graph_type == "networkx_segments":
            self.weight_attr = "cost"
        else:
            self.weight_attr = self.config.get("edge_weight_attr", "length_m")
        self.angle_weight = self.config.get("angle_weight", 1.0)
        self.metrics_save_step = max(
            1,
            int(self.config.get("metrics_save_step", 1)),
        )

    def build_hgraph(self, graph_data: GraphData) -> BaseHGraph:
        """
        Construit un graph en fonction du type déclaré dans la config.
        """
        # Debug allégé : laisser la suite pour le graphe converti uniquement

        graph_type = self.config.get("graph_type", "networkx").lower()
        if graph_type not in {"networkx", "igraph", "networkx_segments"}:
            graph_type = "networkx"

        if graph_type == "igraph":
            graph = build_igraph_graph(graph_data)
        elif graph_type == "networkx_segments":
            graph = build_networkx_segment_graph(graph_data)
            # Calculer le coût pour chaque arête
            self._compute_segment_edge_costs(graph)
        else:
            graph = build_networkx_graph(graph_data)

        if graph_type == "igraph":
            num_nodes = graph.vcount()
            num_edges = graph.ecount()
            is_connected = graph.is_connected()
        else:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            is_connected = nx.is_connected(graph)

        logger.info(
            f"Graphe converti: type={graph_type}, "
            f"nodes={num_nodes}, edges={num_edges}, "
            f"connexe={is_connected}"
        )

        return GraphSimple(
            graph=graph,
            df_cables=graph_data.df_cables,
            nodes_coordinates=graph_data.nodes_coordinates,
            m_per_unit=graph_data.m_per_unit,
            config={"graph_type": graph_type},
        )

    def _compute_segment_edge_costs(self, graph: nx.Graph) -> None:
        """
        Calcule le coût pour chaque arête du graphe de segments.

        Le coût = somme des longueurs des deux nœuds (segments) +
                  angle (abs_cosine) * poids
        """
        for seg1, seg2 in graph.edges():
            if (
                "length_m" not in graph.nodes[seg1]
                or "length_m" not in graph.nodes[seg2]
            ):
                raise ValueError(
                    "Graphe de segments invalide: "
                    "length_m manquant sur un nœud."
                )
            length_seg1 = graph.nodes[seg1].get("length_m")
            length_seg2 = graph.nodes[seg2].get("length_m")
            abs_cosine = graph[seg1][seg2].get("abs_cosine", None)
            if abs_cosine is None:
                raise ValueError(
                    "Graphe de segments invalide: "
                    "abs_cosine manquant sur une arête."
                )

            # Coût = somme des longueurs
            cost = length_seg1 + length_seg2

            # Ajouter la pénalité d'angle si abs_cosine est disponible
            if abs_cosine is not None:
                if abs_cosine < 0 or abs_cosine > 1:
                    logger.warning(
                        f"abs_cosine hors [0,1] pour edge {seg1}-{seg2}: "
                        f"{abs_cosine}"
                    )
                cost += (1-abs_cosine) * float(self.angle_weight)

            graph[seg1][seg2]["cost"] = cost

    def _initialize_run_metrics(self) -> Dict[str, list]:
        """
        Initialise les métriques d'exécution comme dictionnaire de listes.

        Returns:
            Dictionnaire avec les listes de métriques
        """
        return {
            "length_m": [],
            "cost": [],
            "incomplete": [],
            "objective_cost": [],
        }

    def _run_shortest_path(
        self,
        graph: nx.Graph | ig.Graph,
        graph_type: str,
        source: int,
        target: int,
        mode: str,
        nodes_coordinates: List[List[float]],
    ) -> List[int]:
        """
        Utilise NetworkX ou IGraph et retourne la liste des nœuds du chemin.

        Args:
            graph: Le graphe NetworkX ou IGraph
            graph_type: Type de graphe ("networkx", "igraph" ou
                "networkx_segments")
            source: Nœud source
            target: Nœud cible
            mode: Mode de résolution ("dijkstra" ou "a_star")

        Returns:
            Liste des nœuds formant le chemin de source à target
        """
        if source == target:
            return []

        # Cas spécial pour le graphe de segments
        if graph_type == "networkx_segments":
            return self._run_segment_graph_path(
                graph, source, target, mode, nodes_coordinates
            )

        # Vérifier que les nœuds existent dans le graphe
        if graph_type == "networkx":
            if source not in graph or target not in graph:
                return []
        else:
            if source >= graph.vcount() or target >= graph.vcount():
                return []

        # Obtenir le chemin (liste de nœuds)
        try:
            if graph_type == "networkx":
                if mode == "a_star":
                    def heuristic_func(n: int, t: int) -> float:
                        return self._heuristic(n, t, nodes_coordinates)
                    path = nx.astar_path(
                        graph,
                        source,
                        target,
                        heuristic=heuristic_func,
                        weight=self.weight_attr,
                    )
                    return [int(node) for node in path]
                else:
                    path = nx.dijkstra_path(
                        graph,
                        source,
                        target,
                        weight=self.weight_attr,
                    )
                    return [int(node) for node in path]
            else:
                if mode == "a_star":
                    raise ValueError("A* non supporté avec igraph")

                weights = (
                    [float(w) for w in graph.es[self.weight_attr]]
                    if self.weight_attr in graph.es.attributes()
                    else None
                )
                vpaths = graph.get_shortest_paths(
                    v=source,
                    to=target,
                    weights=weights,
                    output="vpath",
                )
                return [int(node) for node in vpaths[0]] if vpaths else []
        except (nx.NetworkXNoPath, Exception):
            return []

    def _run_segment_graph_path(
        self,
        graph: nx.Graph,
        source: int,
        target: int,
        mode: str,
        nodes_coordinates: List[List[float]],
    ) -> List[int]:
        """
        Trouve le plus court chemin dans un graphe de segments.

        Args:
            graph: Graphe de segments (nœuds = tuples (i, j))
            source: Nœud source
            target: Nœud cible
            mode: Mode de résolution ("dijkstra" ou "a_star")

        Returns:
            Liste des nœuds formant le chemin de source à target
        """
        # Trouver tous les segments contenant le nœud source
        source_segments = [
            seg for seg in graph.nodes() if source in seg
        ]

        # Trouver tous les segments contenant le nœud target
        target_segments = [
            seg for seg in graph.nodes() if target in seg
        ]

        if not source_segments or not target_segments:
            return []

        # Si un segment contient à la fois source et target
        common_segments = [
            seg for seg in source_segments if seg in target_segments
        ]
        if common_segments:
            # Le chemin est simplement [source, target]
            return [source, target]

        # Chercher le plus court chemin entre segments
        best_path_segments = None
        best_cost = float("inf")

        for seg_source in source_segments:
            for seg_target in target_segments:
                try:
                    if mode == "a_star":
                        path_segments = nx.astar_path(
                            graph,
                            seg_source,
                            seg_target,
                            heuristic=None,
                            weight="cost",
                        )
                    else:
                        path_segments = nx.dijkstra_path(
                            graph,
                            seg_source,
                            seg_target,
                            weight="cost",
                        )

                    # Calculer le coût total du chemin
                    cost = 0.0
                    for i in range(len(path_segments) - 1):
                        seg1 = path_segments[i]
                        seg2 = path_segments[i + 1]
                        cost += graph[seg1][seg2]["cost"]

                    if cost < best_cost:
                        best_cost = cost
                        best_path_segments = path_segments

                except (nx.NetworkXNoPath, Exception):
                    continue

        if best_path_segments is None:
            return []

        # Convertir le chemin de segments en chemin de nœuds
        return self._convert_segment_path_to_node_path(
            best_path_segments, source, target
        )

    def _convert_segment_path_to_node_path(
        self,
        segment_path: List[tuple],
        source: int,
        target: int,
    ) -> List[int]:
        """
        Convertit un chemin de segments en chemin de nœuds.

        Args:
            segment_path: Liste de segments [(i1, j1), (i2, j2), ...]
            source: Nœud source
            target: Nœud cible

        Returns:
            Liste des nœuds formant le chemin
        """
        if not segment_path:
            return []

        # Reconstruire le chemin de nœuds
        if len(segment_path) == 1:
            # Un seul segment : chemin direct
            seg = segment_path[0]
            if source in seg and target in seg:
                return [source, target]
            return []

        node_path = []
        current_node = source

        for i, seg in enumerate(segment_path):
            # Pour le premier segment, commencer par source
            if i == 0:
                if source not in seg:
                    return []
                # Trouver l'autre nœud du segment (pas source)
                other_node = seg[0] if seg[1] == source else seg[1]
                node_path.extend([source, other_node])
                current_node = other_node
            else:
                # Pour les segments suivants, trouver le nœud commun
                if current_node not in seg:
                    return []
                # Trouver l'autre nœud du segment (pas current_node)
                other_node = seg[0] if seg[1] == current_node else seg[1]
                node_path.append(other_node)
                current_node = other_node

        # Vérifier que le dernier nœud est target
        if node_path[-1] != target:
            # Si target est dans le dernier segment, l'ajouter
            last_seg = segment_path[-1]
            if target in last_seg and current_node != target:
                node_path.append(target)

        return node_path

    def _heuristic(
        self, node: int, target: int, nodes_coordinates: List[List[float]]
    ) -> float:
        """
        Heuristique euclidienne basée sur les positions des nœuds.
        nodes_coordinates est de shape (max_node_id+1, 2) - seulement (x, y).
        """
        if (
            len(nodes_coordinates) == 0
            or node >= len(nodes_coordinates)
            or target >= len(nodes_coordinates)
        ):
            return 0.0
        pos_node = nodes_coordinates[node]
        pos_target = nodes_coordinates[target]
        dx = pos_node[0] - pos_target[0]
        dy = pos_node[1] - pos_target[1]
        return math.sqrt(dx * dx + dy * dy)

    async def _solve_get_var_and_running_metrics(
        self, graph_data: GraphData, **kwargs: Any
    ) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
        """
        Parcourt les câbles, calcule leur chemin et retourne var + run metrics.
        """
        # Construire hgraph comme variable locale
        hgraph = self.build_hgraph(graph_data)

        solve_mode = self.config.get("solve_mode", "dijkstra").lower()
        if solve_mode not in {"dijkstra", "a_star"}:
            solve_mode = "dijkstra"

        graph_type = self.config.get("graph_type", "networkx").lower()
        df_cables_all = hgraph.df_cables
        # has_equipment_loc False ou tenant/aboutissant NaN -> ignorer
        if "has_equipment_loc" in df_cables_all.columns:
            has_loc_series = df_cables_all["has_equipment_loc"].fillna(False)
        else:
            # Si la colonne n'existe pas, tous sont considérés localisés
            has_loc_series = pd.Series(
                [True] * len(df_cables_all),
                index=df_cables_all.index
            )
        missing_loc_mask = (
            (~has_loc_series)
            | df_cables_all["tenant"].isna()
            | df_cables_all["aboutissant"].isna()
        )
        df_missing_loc = df_cables_all[missing_loc_mask]
        df_cables_filtered = df_cables_all[~missing_loc_mask]
        # Trier par cost_per_m si disponible, sinon garder l'ordre original
        if "cost_per_m" in df_cables_filtered.columns:
            df_cables = df_cables_filtered.sort_values(
                by="cost_per_m",
                ascending=False,
            )
        else:
            df_cables = df_cables_filtered

        # Vérifier la connexité du graphe
        if graph_type == "networkx_segments":
            num_nodes = hgraph.graph.number_of_nodes()
            num_edges = hgraph.graph.number_of_edges()
            is_connected = nx.is_connected(hgraph.graph)
        elif graph_type == "networkx":
            num_nodes = hgraph.graph.number_of_nodes()
            num_edges = hgraph.graph.number_of_edges()
            is_connected = nx.is_connected(hgraph.graph)
        else:
            num_nodes = hgraph.graph.vcount()
            num_edges = hgraph.graph.ecount()
            is_connected = hgraph.graph.is_connected()

        logger.info(
            f"Graphe: {num_nodes} nœuds, {num_edges} arêtes, "
            f"connexe: {is_connected}"
        )

        num_cables = len(df_cables)
        num_missing_loc = len(df_missing_loc)
        self.metadata.update(
            {
                "missing_loc_cables": num_missing_loc,
                "total_cables_input": len(df_cables_all),
            }
        )
        runs_metrics = self._initialize_run_metrics()

        incomplete_cables = 0
        if num_missing_loc > 0:
            logger.warning(
                f"Câbles ignorés faute de localisation: {num_missing_loc} "
                f"/ {len(df_cables_all)}"
            )
            for cable_idx in df_missing_loc.index:
                hgraph.update_var({cable_idx: []})
                incomplete_cables += 1

        sequential_idx = 0

        for cable_idx, row in tqdm(
            df_cables.iterrows(),
            total=num_cables,
            desc="Résolution des câbles",
        ):
            source = int(row["tenant"])
            target = int(row["aboutissant"])

            path_nodes = self._run_shortest_path(
                graph=hgraph.graph,
                graph_type=graph_type,
                source=source,
                target=target,
                mode=solve_mode,
                nodes_coordinates=hgraph.nodes_coordinates,
            )

            if not path_nodes:
                incomplete_cables += 1
                logger.warning(
                    f"Câble {cable_idx}: Aucun chemin trouvé entre "
                    f"tenant={source} et aboutissant={target}"
                )

            hgraph.update_var({cable_idx: path_nodes})

            if sequential_idx % self.metrics_save_step == 0:
                df_metrics_step = shortest_path_running_metrics(
                    df_cables=hgraph.df_cables,
                    var=hgraph.var,
                    nodes_coordinates=hgraph.nodes_coordinates,
                    m_per_unit=hgraph.m_per_unit,
                )
                total_length = df_metrics_step["metric_length_m"].sum()
                total_cost = df_metrics_step["metric_cost"].sum()
                incomplete_count = int(
                    df_metrics_step["metric_is_uncomplete_path"].sum()
                )
                objective_cost = compute_objective_from_simplehgraph(
                    hgraph
                )

                runs_metrics["length_m"].append(float(total_length))
                runs_metrics["cost"].append(float(total_cost))
                runs_metrics["incomplete"].append(incomplete_count)
                runs_metrics["objective_cost"].append(float(objective_cost))

            sequential_idx += 1

        logger.info(f"Câbles incomplets: {incomplete_cables}/{num_cables}")
        return hgraph.var, runs_metrics


def test_shortest_path_solver_pipeline(
    graph_type: str = "networkx",
    solve_mode: str = "dijkstra",
) -> None:
    """
    Test minimal pour le solveur de plus court chemin.
    """
    import pickle
    from pathlib import Path

    # Charger les données depuis le cache si disponible
    # Chercher le fichier depuis le répertoire du script
    script_file = Path(__file__).resolve()
    script_dir = script_file.parent
    project_root = script_dir.parent
    cache_path = project_root / "Data" / "graph_data_cache.pkl"
    
    # Vérifier que le fichier existe
    if not cache_path.exists():
        print(f"Cache non trouvé à: {cache_path}")
        print(f"Script: {script_file}")
        print(f"Racine projet: {project_root}")
        print(f"Vérification de l'existence du dossier Data: {(project_root / 'Data').exists()}")
        return
    
    print(f"Chargement du cache: {cache_path}")
    try:
        with open(cache_path, "rb") as f:
            graph_data = pickle.load(f)
        print(f"Données chargées avec succès: {len(graph_data.df_cables)} câbles")
    except Exception as e:
        print(f"Erreur lors du chargement du cache: {e}")
        import traceback
        traceback.print_exc()
        return

    solver = ShortestPathSolver(
        config={
            "graph_type": graph_type,
            "solve_mode": solve_mode,
            "edge_weight_attr": "length_m",
            "metrics_save_step": 10,
            "angle_weight": 1e2,
            "m_per_unit": 1,
        },
    )

    import asyncio
    solver_result = asyncio.run(
        solver.solve(
            graph_data=graph_data,
            compute_metrics=True,
        )
    )

    print("\n--- Résultats du solveur ---")
    if solver_result.eval_metrics:
        print(f"Temps de calcul: {solver_result.eval_metrics.computation_time:.2f} s")
        print(f"Câbles complets: {solver_result.eval_metrics.num_complete_cables}/{solver_result.eval_metrics.total_cables}")
        print(f"Couverture: {solver_result.eval_metrics.coverage_pct:.2f}%")


if __name__ == "__main__":
    # add tests
    test_shortest_path_solver_pipeline(
        graph_type="networkx",
        solve_mode="dijkstra",
    )
