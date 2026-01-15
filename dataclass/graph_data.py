# -*- coding: utf-8 -*-
############
# Packages #
############
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))
#######################
# Internal Imports #
#######################

#########
# Utils #
#########


@dataclass
class GraphData:
    """
    Format pivot utilisant GeoPandas pour unifier géométries et relations.

    gdf_segments: GeoDataFrame avec géométries LineString, colonnes i, j,
                  length_m
    gdf_nodes: GeoDataFrame avec géométries Point et colonne node_id
    nodes_coordinates: np.ndarray de shape (max_node_id+1, 2) où l'index
                       correspond au node_id
    df_cables: DataFrame avec colonnes tenant, aboutissant (node_id),
               tenant_name, aboutissant_name (noms d'équipements d'origine),
               length_gt (GT), has_equipment_loc (bool localisation trouvée)
    df_angles: DataFrame avec colonnes i, j, k, abs_cosine
    m_per_unit: Facteur de conversion des unités DXF vers mètres
    metadata: Dictionnaire pour informations globales
    """

    gdf_segments: gpd.GeoDataFrame
    gdf_nodes: gpd.GeoDataFrame
    nodes_coordinates: np.ndarray
    df_cables: pd.DataFrame
    df_angles: pd.DataFrame
    m_per_unit: float
    metadata: Dict[str, Any] = field(default_factory=dict)


if __name__ == "__main__":
    # add tests
    pass
