# -*- coding: utf-8 -*-
############
# Packages #
############
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).parents[1]))

#######################
# Internal Imports #
#######################
from dataclass.graph_data import GraphData
from dataclass.graph_data_with_var import GraphDataWithVar
#########
# Utils #
#########


@dataclass
class EvalMetrics:
    """
    Métriques d'évaluation standardisées pour tous les solveurs.

    Attributes:
        computation_time: Temps de calcul en secondes
        num_complete_cables: Nombre de câbles avec chemins complets
        total_cables: Nombre total de câbles
        coverage_pct: Pourcentage de couverture (câbles complets / total)
        total_length_pred_m: Longueur totale prédite (complets uniquement)
        total_length_gt_m: Longueur GT totale (TOUS les câbles,
            même incomplets)
        total_weight_pred_kg: Poids total prédit en kilogrammes
            (complets uniquement, via Poids_kg_per_km)
        total_weight_gt_kg: Poids total GT en kilogrammes (tous les
            câbles) si Poids_kg_per_km est disponible
        mean_length_pred_m: Longueur moyenne prédite (complets uniquement)
        mean_length_gt_m: Longueur moyenne GT (complets uniquement)
        std_length_pred_m: Écart-type longueur prédite (complets uniquement)
        std_length_gt_m: Écart-type longueur GT (complets uniquement)
        total_length_pred_loc_m: Longueur prédite (complets ET localisés)
        total_length_gt_loc_m: Longueur GT (complets ET localisés)
        missing_loc_cables: Nombre de câbles sans localisation
        num_right_angles_with_tolerance: Nombre d'angles droits (80-100°)
        total_angles: Nombre total d'angles
        diff_total_m: Différence totale (pred - gt, complets uniquement)
        diff_mean_m: Différence moyenne (pred - gt, complets uniquement)
        diff_median_m: Différence médiane (pred - gt, complets uniquement)
        diff_std_m: Écart-type des différences (complets uniquement)
    """

    # Ordre d'affichage et labels pour les métriques
    _display_order = [
        ("computation_time", "Temps de calcul", "{:.2f} s"),
        ("total_cables", "Total câbles", "{}"),
        ("num_complete_cables", "Câbles complets", "{}"),
        # Note: "Câbles incomplets" est calculé dynamiquement dans le plot
        ("coverage_pct", "Couverture", "{:.2f}%"),
        ("total_length_pred_m", "Longueur totale prédite", "{:.2f} m"),
        ("total_length_gt_m", "Longueur totale GT", "{:.2f} m"),
        ("total_weight_pred_kg", "Poids total prédit", "{:.2f} kg"),
        ("total_weight_gt_kg", "Poids total GT", "{:.2f} kg"),
        ("mean_length_pred_m", "Longueur moyenne prédite", "{:.2f} m"),
        ("mean_length_gt_m", "Longueur moyenne GT", "{:.2f} m"),
        ("std_length_pred_m", "Écart-type longueur prédite", "{:.2f} m"),
        ("std_length_gt_m", "Écart-type longueur GT", "{:.2f} m"),
        ("total_length_pred_loc_m", "Longueur prédite (loc.)", "{:.2f} m"),
        ("total_length_gt_loc_m", "Longueur GT (loc.)", "{:.2f} m"),
        ("missing_loc_cables", "Câbles sans localisation", "{}"),
        (
            "num_right_angles_with_tolerance",
            "Angles droits (80-100°)",
            "{}"
        ),
        ("total_angles", "Total angles", "{}"),
        ("diff_total_m", "Différence totale", "{:.2f} m"),
        ("diff_mean_m", "Différence moyenne", "{:.2f} m"),
        ("diff_median_m", "Différence médiane", "{:.2f} m"),
        ("diff_std_m", "Écart-type différences", "{:.2f} m"),
    ]

    computation_time: float
    num_complete_cables: int
    total_cables: int
    coverage_pct: float
    total_length_pred_m: float
    total_length_gt_m: float
    total_weight_pred_kg: float
    total_weight_gt_kg: float
    mean_length_pred_m: float
    mean_length_gt_m: float
    std_length_pred_m: float
    std_length_gt_m: float
    total_length_pred_loc_m: float
    total_length_gt_loc_m: float
    missing_loc_cables: int
    num_right_angles_with_tolerance: int
    total_angles: int
    diff_total_m: float
    diff_mean_m: float
    diff_median_m: float
    diff_std_m: float

    def to_display_dict(self) -> Dict[str, str]:
        """
        Convertit les métriques en dictionnaire ordonné pour l'affichage.

        Returns:
            Dictionnaire {label: valeur_formatée} dans l'ordre défini
        """
        result = {}
        for attr_name, label, format_str in self._display_order:
            value = getattr(self, attr_name)
            result[label] = format_str.format(value)
        return result


@dataclass
class SolverResult:
    """
    Format pivot pour standardiser la sortie des solveurs.

    Cette classe encapsule tous les résultats d'un solveur de manière
    standardisée, permettant l'utilisation d'un bloc d'évaluation unique.

    Attributes:
        graph_data: Données du graphe (segments, nœuds, câbles)
        cable_paths: Dictionnaire {cable_idx: [node1, node2, ...]} avec les
                     chemins résolus pour chaque câble
        eval_metrics: Métriques d'évaluation standardisées
        runs_metrics: Métriques spécifiques au modèle (flexible, dépendant
                      du solveur). Exemples :
                      - Pour Q-Learning: episode_rewards, training_steps
                      - Pour Shortest Path: metrics_tensor, cost_tensor
        model_id: Identifiant du modèle/solveur
        metadata: Métadonnées additionnelles (optionnel)
    """

    graph_data: GraphDataWithVar
    cable_paths: Dict[int, list[int]]
    eval_metrics: EvalMetrics
    model_id: str
    runs_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


if __name__ == "__main__":
    # add tests
    pass
