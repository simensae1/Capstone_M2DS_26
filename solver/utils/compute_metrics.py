# -*- coding: utf-8 -*-
############
# Packages #
############
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).parents[2]))

#######################
# Internal Imports #
#######################
from dataclass.graph_data_with_var import GraphDataWithVar
from dataclass.solver_results import EvalMetrics

#########
# Utils #
#########


def _stats_from_series(series: pd.Series) -> Dict[str, float]:
    """Retourne sum/mean/median/std avec gestion des séries vides."""
    if series.empty:
        return {"sum": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "sum": float(series.sum()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
    }


###########
# metrics #
###########

def _length_m(
    df_cables: pd.DataFrame,
    nodes_coordinates: np.ndarray,
    var: Dict[int, List[int]],
    m_per_unit: float,
) -> np.ndarray:
    """
    Calcule la longueur de chaque câble en mètres.

    Args:
        df_cables: DataFrame des câbles
        nodes_coordinates: Coordonnées des nœuds (n_nodes, 3)
        var: Dictionnaire {cable_idx: [node1, node2, ...]}
        m_per_unit: Facteur de conversion en mètres

    Returns:
        Array des longueurs en mètres
    """
    length_m = np.zeros(len(df_cables))
    for cable_idx, path_nodes in var.items():
        if path_nodes and len(path_nodes) > 1:
            path_coords = nodes_coordinates[path_nodes]
            # Calculer les distances entre nœuds consécutifs
            dist = path_coords[1:] - path_coords[:-1]
            # Calculer la norme euclidienne de chaque segment
            segment_lengths = np.linalg.norm(dist, axis=1)
            # Somme totale et conversion en mètres
            length_m[cable_idx] = np.sum(segment_lengths) * m_per_unit
    return length_m


def _weight_kg(
    df_cables: pd.DataFrame,
    length_m: np.ndarray,
) -> np.ndarray:
    """Compute cable weight in kilograms using Poids_kg_per_km when present."""
    if "Poids_kg_per_km" not in df_cables.columns:
        return np.zeros(len(df_cables))

    poids_per_km = pd.to_numeric(
        df_cables["Poids_kg_per_km"], errors="coerce"
    ).fillna(0.0)
    poids_per_m = poids_per_km / 1000.0
    return length_m * poids_per_m.to_numpy()


def _cost(
    df_cables: pd.DataFrame,
    length_m: np.ndarray,
) -> np.ndarray:
    """
    Calcule le coût de chaque câble.

    Args:
        df_cables: DataFrame des câbles
        length_m: Longueurs en mètres

    Returns:
        Array des coûts
    """
    return length_m * df_cables["cost_per_m"].values


def _is_uncomplete_path(
    df_cables: pd.DataFrame,
    var: Dict[int, List[int]],
) -> np.ndarray:
    """
    Vérifie si chaque câble a un chemin complet (atteint son aboutissant).

    Args:
        df_cables: DataFrame des câbles
        var: Dictionnaire {cable_idx: [node1, node2, ...]}

    Returns:
        Array de 0/1 (1 = chemin incomplet, 0 = chemin complet)
    """
    incomplete = np.zeros(len(df_cables), dtype=int)
    for idx, row in df_cables.iterrows():
        target_val = row["aboutissant"]
        path_nodes = var.get(idx, [])

        # Si tenant/aboutissant manquants -> chemin incomplet
        if pd.isna(target_val):
            incomplete[idx] = 1
            continue

        target = int(target_val)
        # Chemin incomplet si le nœud aboutissant n'est pas dans le chemin
        incomplete[idx] = 1 if target not in path_nodes else 0
    return incomplete


def _compute_angle_cosine(
    nodes_coordinates: np.ndarray,
    i_idx: int,
    j_idx: int,
    k_idx: int,
) -> Optional[float]:
    """
    Calcule le cosinus de l'angle formé par trois nœuds consécutifs.

    Args:
        nodes_coordinates: Coordonnées des nœuds (n_nodes, 3)
        i_idx: Index du premier nœud
        j_idx: Index du nœud central
        k_idx: Index du dernier nœud

    Returns:
        Cosinus de l'angle ou None si invalide
    """
    # Vérifier les indices valides
    if (
        i_idx >= len(nodes_coordinates)
        or j_idx >= len(nodes_coordinates)
        or k_idx >= len(nodes_coordinates)
    ):
        return None

    # Vecteurs directionnels
    v1 = nodes_coordinates[j_idx] - nodes_coordinates[i_idx]
    v2 = nodes_coordinates[k_idx] - nodes_coordinates[j_idx]

    # Calculer le cosinus de l'angle
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 > 1e-10 and norm_v2 > 1e-10:
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

    return None


def _count_angles(
    df_cables: pd.DataFrame,
    var: Dict[int, List[int]],
    nodes_coordinates: np.ndarray,
) -> np.ndarray:
    """
    Calcule le nombre d'angles pour chaque câble.
    Un angle est formé par trois nœuds consécutifs (i, j, k) dans le chemin.

    Args:
        df_cables: DataFrame des câbles
        var: Dictionnaire {cable_idx: [node1, node2, ...]}
        nodes_coordinates: Coordonnées des nœuds (n_nodes, 3)

    Returns:
        Array avec le nombre d'angles par câble
    """
    angle_counts = np.zeros(len(df_cables), dtype=int)

    for cable_idx, path_nodes in var.items():
        if cable_idx >= len(df_cables) or len(path_nodes) < 3:
            continue

        # Compter les triplets consécutifs (i, j, k)
        angle_counts[cable_idx] = len(path_nodes) - 2

    return angle_counts


def _count_angles_between_range(
    df_cables: pd.DataFrame,
    var: Dict[int, List[int]],
    nodes_coordinates: np.ndarray,
    min_angle_deg: float = 80.0,
    max_angle_deg: float = 100.0,
) -> np.ndarray:
    """
    Calcule le nombre d'angles dans une plage donnée (en degrés).
    Un angle est formé par trois nœuds consécutifs (i, j, k) dans le chemin.

    Args:
        df_cables: DataFrame des câbles
        var: Dictionnaire {cable_idx: [node1, node2, ...]}
        nodes_coordinates: Coordonnées des nœuds (n_nodes, 3)
        min_angle_deg: Angle minimum en degrés (défaut: 80°)
        max_angle_deg: Angle maximum en degrés (défaut: 100°)

    Returns:
        Array avec le nombre d'angles dans la plage par câble
    """
    angle_counts = np.zeros(len(df_cables), dtype=int)

    # Convertir les angles en seuils de cosinus
    min_cosine = np.cos(np.radians(max_angle_deg))
    max_cosine = np.cos(np.radians(min_angle_deg))

    for cable_idx, path_nodes in var.items():
        if cable_idx >= len(df_cables) or len(path_nodes) < 3:
            continue

        count = 0
        # Parcourir tous les triplets consécutifs
        for i in range(len(path_nodes) - 2):
            cosine = _compute_angle_cosine(
                nodes_coordinates,
                path_nodes[i],
                path_nodes[i + 1],
                path_nodes[i + 2],
            )

            if cosine is not None and min_cosine <= cosine <= max_cosine:
                count += 1

        angle_counts[cable_idx] = count

    return angle_counts

###################
# Running Metrics #
###################


def shortest_path_running_metrics(
    df_cables: pd.DataFrame,
    var: Dict[int, List[int]],
    nodes_coordinates: np.ndarray,
    m_per_unit: float,
    cable_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    API unique : ajoute les colonnes metric_* directement dans df_cables.
    """
    var_filtered = (
        {idx: var.get(idx, []) for idx in cable_indices}
        if cable_indices is not None
        else var
    )
    df_cables["metric_length_m"] = _length_m(
        df_cables, nodes_coordinates, var_filtered, m_per_unit
    )
    df_cables["metric_cost"] = _cost(df_cables, df_cables["metric_length_m"])
    df_cables["metric_is_uncomplete_path"] = _is_uncomplete_path(
        df_cables, var_filtered
    )
    df_cables["metric_weight_kg"] = _weight_kg(
        df_cables, df_cables["metric_length_m"]
    )
    df_cables["metric_num_angles"] = _count_angles(
        df_cables, var_filtered, nodes_coordinates
    )
    df_cables["metric_num_angles_within_tolerance_10pct"] = (
        _count_angles_between_range(
            df_cables,
            var_filtered,
            nodes_coordinates,
            min_angle_deg=80.0,
            max_angle_deg=100.0,
        )
    )
    return df_cables


################
# Eval Metrics #
################

def compute_eval_metrics(
    graph_data_with_var: GraphDataWithVar,
    computation_time: float,
    length_col_gt: str = "length_gt",
) -> EvalMetrics:
    """
    Calcule EvalMetrics directement depuis GraphDataWithVar.
    Ajoute la colonne persistante length_diff_with_gt dans df_cables.
    """
    df_cables = graph_data_with_var.df_cables  # pas de copie: persiste

    shortest_path_running_metrics(
        df_cables=df_cables,
        var=graph_data_with_var.var,
        nodes_coordinates=graph_data_with_var.nodes_coordinates,
        m_per_unit=graph_data_with_var.m_per_unit,
    )

    df_cables["length_diff_with_gt"] = (
        df_cables["metric_length_m"] - df_cables[length_col_gt]
        if length_col_gt in df_cables.columns
        else np.nan
    )

    mask_complete = df_cables["metric_is_uncomplete_path"] == 0
    num_complete_cables = int(mask_complete.sum())
    total_cables = len(df_cables)
    coverage_pct = (
        (num_complete_cables / total_cables * 100.0)
        if total_cables > 0
        else 0.0
    )
    # Log mask_complete supprimé (trop verbeux, déjà dans stats finales)

    mask_loc = (
        df_cables["has_equipment_loc"].fillna(False)
        if "has_equipment_loc" in df_cables.columns
        else pd.Series(False, index=df_cables.index)
    )

    lengths_pred_complete = df_cables.loc[mask_complete, "metric_length_m"]
    stats_pred = _stats_from_series(lengths_pred_complete)

    has_gt = length_col_gt in df_cables.columns
    lengths_gt_complete = (
        df_cables.loc[mask_complete, length_col_gt]
        if has_gt
        else pd.Series(dtype=float)
    )
    stats_gt = _stats_from_series(lengths_gt_complete) if has_gt else {
        "sum": 0.0,
        "mean": 0.0,
        "median": 0.0,
        "std": 0.0,
    }
    total_length_gt_m = (
        float(df_cables[length_col_gt].sum()) if has_gt else 0.0
    )

    mask_complete_loc = mask_complete & mask_loc
    total_length_pred_loc_m = float(
        df_cables.loc[mask_complete_loc, "metric_length_m"].sum()
    )
    total_length_gt_loc_m = (
        float(df_cables.loc[mask_complete_loc, length_col_gt].sum())
        if has_gt
        else 0.0
    )

    missing_loc_cables = int((~mask_loc).sum()) if len(mask_loc) else 0
    num_right_angles_with_tolerance = int(
        df_cables["metric_num_angles_within_tolerance_10pct"].sum()
        if "metric_num_angles_within_tolerance_10pct" in df_cables.columns
        else 0
    )
    total_angles = int(
        df_cables["metric_num_angles"].sum()
        if "metric_num_angles" in df_cables.columns
        else 0
    )

    total_weight_pred_kg = float(
        df_cables.loc[mask_complete, "metric_weight_kg"].sum()
    ) if "metric_weight_kg" in df_cables.columns else 0.0
    if has_gt and "Poids_kg_per_km" in df_cables.columns:
        poids_per_m = pd.to_numeric(
            df_cables["Poids_kg_per_km"], errors="coerce"
        ).fillna(0.0) / 1000.0
        # Calcul du poids GT sur le subset complet
        lengths_gt_masked = df_cables.loc[mask_complete, length_col_gt]
        poids_per_m_masked = poids_per_m.loc[mask_complete]
        total_weight_gt_kg = float(
            (lengths_gt_masked * poids_per_m_masked).sum()
        )
    else:
        total_weight_gt_kg = 0.0

    if (
        has_gt
        and len(lengths_pred_complete) > 0
        and len(lengths_gt_complete) > 0
    ):
        diff_series = lengths_pred_complete - lengths_gt_complete
        stats_diff = _stats_from_series(diff_series)
    else:
        stats_diff = {
            "sum": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
        }

    total_length_pred_m = stats_pred["sum"]
    mean_length_pred_m = stats_pred["mean"]
    std_length_pred_m = stats_pred["std"]
    mean_length_gt_m = stats_gt["mean"]
    std_length_gt_m = stats_gt["std"]
    diff_total_m = stats_diff["sum"]
    diff_mean_m = stats_diff["mean"]
    diff_median_m = stats_diff["median"]
    diff_std_m = stats_diff["std"]

    return EvalMetrics(
        computation_time=computation_time,
        num_complete_cables=num_complete_cables,
        total_cables=total_cables,
        coverage_pct=coverage_pct,
        total_length_pred_m=total_length_pred_m,
        total_length_gt_m=total_length_gt_m,
        mean_length_pred_m=mean_length_pred_m,
        mean_length_gt_m=mean_length_gt_m,
        std_length_pred_m=std_length_pred_m,
        std_length_gt_m=std_length_gt_m,
        total_length_pred_loc_m=total_length_pred_loc_m,
        total_length_gt_loc_m=total_length_gt_loc_m,
        missing_loc_cables=missing_loc_cables,
        total_weight_pred_kg=total_weight_pred_kg,
        total_weight_gt_kg=total_weight_gt_kg,
        num_right_angles_with_tolerance=num_right_angles_with_tolerance,
        total_angles=total_angles,
        diff_total_m=diff_total_m,
        diff_mean_m=diff_mean_m,
        diff_median_m=diff_median_m,
        diff_std_m=diff_std_m,
    )
