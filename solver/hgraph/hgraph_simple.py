# -*- coding: utf-8 -*-
############
# Packages #
############
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[2]))
#######################
# Internal Imports #
#######################
from solver.hgraph.base_hgraph import BaseHGraph

#########
# Utils #
#########


class GraphSimple(BaseHGraph):
    """
    Wrapper unique pour NetworkX ou Igraph, piloté par la config graph_type.
    Cette classe stocke les données nécessaires au calcul des métriques,
    notamment nodes_coordinates qui est utilisé pour calculer la longueur
    des chemins de câbles via les distances euclidiennes entre nœuds.
    """
    def __init__(
        self,
        graph: nx.Graph | ig.Graph,
        df_cables: pd.DataFrame,
        nodes_coordinates: np.ndarray,
        m_per_unit: float,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialise GraphSimple.

        Args:
            graph: Graphe NetworkX ou Igraph
            df_cables: DataFrame des câbles avec colonnes tenant, aboutissant
            nodes_coordinates: Array de shape (max_node_id+1, 2) avec
                             coordonnées (x, y) des nœuds. Utilisé pour
                             calculer les métriques (longueur de chemin) via
                             distances euclidiennes. L'index correspond au
                             node_id.
            m_per_unit: Facteur de conversion des unités vers mètres
            config: Configuration optionnelle
        """
        super().__init__(config=config)
        self.graph = graph
        self.df_cables = df_cables
        self.nodes_coordinates = nodes_coordinates
        self.m_per_unit = m_per_unit
        self.var: Dict[int, List[int]] = {}
        self.graph_type = (
            "networkx" if isinstance(graph, nx.Graph) else "igraph"
        )

    def get_var(self) -> Dict[str, Any]:
        """Retourne les variables à optimiser."""
        return self.var

    def update_var(self, update_dict: Dict[str, Any]) -> None:
        """
        Met à jour les variables à optimiser.

        Args:
            update_dict: Dictionnaire {cable_idx: path_nodes} où path_nodes
                        est une liste de nœuds formant le chemin
        """
        for cable_idx, path_nodes in update_dict.items():
            self.var[int(cable_idx)] = [int(node) for node in path_nodes]

    def get_subgraph(self, filters: Dict[str, Any]) -> Any:
        """
        Retourne un sous-graphe selon des filtres.
        """
        if not filters:
            return self.graph.copy()

        nodes = filters.get("nodes")
        edges = filters.get("edges")
        if self.graph_type == "networkx":
            if nodes:
                return self.graph.subgraph(list(nodes)).copy()
            if edges:
                return self.graph.edge_subgraph(list(edges)).copy()
            return self.graph.copy()

        if nodes:
            indices = [int(node) for node in nodes]
            return self.graph.induced_subgraph(indices)
        return self.graph.copy()


if __name__ == "__main__":
    # add tests
    pass
