# -*- coding: utf-8 -*-
############
# Packages #
############
import sys
from pathlib import Path

import geopandas as gpd
import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

sys.path.append(str(Path(__file__).parents[2]))

#######################
# Internal Imports #
#######################
from dataclass.graph_data import GraphData

#########
# Utils #
#########


def build_networkx_graph(graph_data: GraphData) -> nx.Graph:
    """
    Construit un graphe NetworkX à partir d'un objet GraphData.
    """
    graph = nx.Graph()
    df_segments = graph_data.gdf_segments[['i', 'j', 'length_m']].copy()
    if df_segments is None or df_segments.empty:
        return graph

    for segment_idx, row in df_segments.iterrows():
        graph.add_edge(
            int(row["i"]),
            int(row["j"]),
            length_m=float(row["length_m"]),
            segment_idx=int(segment_idx),
        )

    if len(graph_data.gdf_nodes) > 0:
        nodes_coordinates = graph_data.nodes_coordinates
        nodes_pos_dict = {}
        for node_id in range(len(nodes_coordinates)):
            coord = nodes_coordinates[node_id]
            nodes_pos_dict[node_id] = tuple(coord)  # (x, y)
        nx.set_node_attributes(graph, nodes_pos_dict, "pos")

    return graph


def build_igraph_graph(graph_data: GraphData) -> ig.Graph:
    """
    Construit un graphe igraph à partir d'un objet GraphData.
    """
    df_segments = graph_data.gdf_segments[['i', 'j', 'length_m']].copy()
    if df_segments is None or df_segments.empty:
        return ig.Graph()

    i_nodes = df_segments["i"].astype(int).tolist()
    j_nodes = df_segments["j"].astype(int).tolist()
    graph = ig.Graph()
    if i_nodes or j_nodes:
        max_node_id = max(i_nodes + j_nodes)
        graph.add_vertices(max_node_id + 1)
    edges = list(zip(i_nodes, j_nodes))
    if edges:
        graph.add_edges(edges)
    if "length_m" in df_segments.columns:
        graph.es["length_m"] = df_segments["length_m"].astype(float).tolist()
    # Stocker l'index du segment dans df_segments
    graph.es["segment_idx"] = df_segments.index.astype(int).tolist()

    return graph


def build_networkx_segment_graph(graph_data: GraphData) -> nx.Graph:
    """
    Construit un graphe NetworkX où chaque nœud représente un segment.

    Dans ce graphe:
    - Les nœuds sont des segments identifiés par (i, j)
    - Les arêtes relient deux segments qui partagent un nœud
      (ex: segment (i,j) et (i,k) ou (i,j) et (j,k))
    - Chaque nœud a l'attribut length_m et pos (moyenne des positions)
    - Chaque arête a l'attribut abs_cosine provenant de df_angles
    """
    graph = nx.Graph()
    df_segments = graph_data.gdf_segments.copy()

    if df_segments is None or df_segments.empty:
        return graph

    if df_segments["length_m"].isna().any():
        nan_rows = df_segments[df_segments["length_m"].isna()]
        raise ValueError("length_m manquant sur au moins un segment")

    nodes_coordinates = graph_data.nodes_coordinates

    # Créer tous les nœuds (segments) en parcourant gdf_segments
    for segment_idx, row in df_segments.iterrows():
        i, j = int(row["i"]), int(row["j"])
        # Calculer la position moyenne pour les plots
        pos_i = nodes_coordinates[i]
        pos_j = nodes_coordinates[j]
        pos_mean = tuple((pos_i + pos_j) / 2.0)

        graph.add_node(
            (i, j),
            length_m=float(row["length_m"]),
            segment_idx=int(segment_idx),
            pos=pos_mean,
        )

    # Créer un set des segments pour vérification rapide
    segments_set = set(graph.nodes())

    if graph_data.df_angles is None:
        raise ValueError("df_angles est None alors que des segments existent.")

    for _, angle_row in graph_data.df_angles.iterrows():
        i = int(angle_row["i"])
        j = int(angle_row["j"])
        k = int(angle_row["k"])
        abs_cosine = float(angle_row["abs_cosine"])
        if pd.isna(abs_cosine):
            raise ValueError(
                f"abs_cosine NaN détecté pour l'angle ({i},{j},{k})"
            )

        # Les segments sont normalisés avec i < j
        seg1 = (min(i, j), max(i, j))
        seg2 = (min(j, k), max(j, k))
        if seg1 != seg2 and seg1 in segments_set and seg2 in segments_set:
            graph.add_edge(seg1, seg2, abs_cosine=abs_cosine)
           
    return graph


def test_igraph_build() -> None:
    """
    Test de base pour la construction Igraph.
    """
    # Créer les géométries LineString pour les segments
    geometries_segments = [
        LineString([(0.0, 0.0), (2.0, 1.0)]),
        LineString([(2.0, 1.0), (2.0, 2.0)]),
    ]
    gdf_segments = gpd.GeoDataFrame(
        {
            "i": [0, 2],
            "j": [2, 3],
            "length_m": [7.0, 4.0],
            "capacity": [120, 60],
        },
        geometry=geometries_segments,
    )

    # Créer les géométries Point pour les nœuds
    geometries_nodes = [
        Point(0.0, 0.0),
        Point(2.0, 0.0),
        Point(2.0, 1.0),
        Point(2.0, 2.0),
    ]
    gdf_nodes = gpd.GeoDataFrame(
        {"node_id": [0, 1, 2, 3]},
        geometry=geometries_nodes,
    )

    # nodes_coordinates est de shape (max_node_id+1, 2)
    nodes_coordinates = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )

    graph_input = GraphData(
        gdf_segments=gdf_segments,
        gdf_nodes=gdf_nodes,
        nodes_coordinates=nodes_coordinates,
        df_cables=pd.DataFrame(),
        df_angles=pd.DataFrame(),
        m_per_unit=1.0,
        metadata={},
    )
    graph = build_igraph_graph(graph_input)
    assert graph.ecount() == 2
    assert graph.es["length_m"] == [7.0, 4.0]
    assert graph.es["length_m"][1] == 4.0





def test_networkx_segment_graph_from_builder(
    excel_id: str = "local_test/raw/carnet_de_cables/pfe_salma",
    autocad_id: str = "local_test/raw/plan_implementation/dxf/pfe_salma",
    layer_filter: str = "ACT--Chemin de cables",
    equipment_layer: str = "Equipement Electrique",
) -> None:
    """
    Charge un GraphData comme dans ShortestPathSolver et visualise le graphe.
    """
    import asyncio
    from datetime import datetime

    from config.paths import PLOTS_PATH
    from src.pipelines_modules.functionals.raw_to_graph_data.graph_data_builder_local import (  # noqa: E501
        GraphDataBuilderLocal,
    )
    from src.pipelines_modules.repositories.local import config_and_data_repo
    from src.utils.plot.network_plot import visualize_networkx_segment_graph

    async def run_test() -> None:
        data_repo = config_and_data_repo.DataRepo()
        df_carnet = data_repo.get(excel_id, filetype="excel")
        dxf_document = data_repo.get(autocad_id, filetype="autocad")
        if df_carnet is None or dxf_document is None:
            print("Données de test indisponibles, test ignoré.")
            return

        builder = GraphDataBuilderLocal(
            config={
                "column_map": {
                    "tenant": "tenant",
                    "aboutissant": "aboutissant",
                    "section": "section",
                    "marque": "marque",
                    "longueur": "length_gt",
                },
                "default_cost": 1.0,
                "data_col_to_keep": [
                    "tenant",
                    "aboutissant",
                    "cost_per_m",
                    "mode",
                    "length_gt",
                    "has_equipment_loc",
                ],
                "layer_filter": layer_filter,
                "equipment_layer": equipment_layer,
                "snap_threshold": 70000.0,
                "m_per_unit": 1.0,
            }
        )
        graph_data = await builder.execute(
            {
                "dxf_document": dxf_document,
                "df_carnet": df_carnet,
                "df_costs": None,
            }
        )
        graph = build_networkx_segment_graph(graph_data)
        fig = visualize_networkx_segment_graph(graph, show_labels=False)

        output_dir = PLOTS_PATH / "tests"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = output_dir / f"segment_graph_test_{timestamp}.html"
        fig.write_html(str(save_path))
        print(f"Plot segment graph sauvegardé: {save_path}")

    asyncio.run(run_test())


if __name__ == "__main__":
    # add tests
    test_networkx_segment_graph_from_builder()
    print("succed")
