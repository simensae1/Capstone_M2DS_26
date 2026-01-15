# -*- coding: utf-8 -*-
############
# Packages #
############
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

sys.path.append(str(Path(__file__).parents[2]))
#######################
# Internal Imports #
#######################
from solver.hgraph.base_hgraph import BaseHGraph
from solver.utils.compute_metrics import shortest_path_running_metrics

#########
# Utils #
#########


def compute_objective_from_simplehgraph(
    hgraph: BaseHGraph,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Agrège les métriques du hgraph pour produire un score pondéré unique.
    """
    metrics_df = shortest_path_running_metrics(
        df_cables=hgraph.df_cables,
        var=hgraph.var,
        nodes_coordinates=hgraph.nodes_coordinates,
        m_per_unit=hgraph.m_per_unit,
    )
    base_weights = {
        "metric_cost": 1.0,
        "metric_is_uncomplete_path": 1.0,
    }
    if weights:
        base_weights.update(weights)

    cost_term = float(metrics_df["metric_cost"].sum())
    incomplete_term = float(
        metrics_df["metric_is_uncomplete_path"].astype(pd.Int64Dtype()).sum()
    )

    objective_value = (
        base_weights["metric_cost"] * cost_term
        + base_weights["metric_is_uncomplete_path"] * incomplete_term
    )
    return objective_value


if __name__ == "__main__":
    # add tests
    pass
