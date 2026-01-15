# -*- coding: utf-8 -*-
############
# Packages #
############
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from abc import ABC, abstractmethod

sys.path.append(str(Path(__file__).parents[1]))
#######################
# Internal Imports #
#######################
from dataclass.graph_data import GraphData
from dataclass.graph_data_with_var import GraphDataWithVar
from dataclass.solver_results import SolverResult, EvalMetrics
from solver.utils.compute_metrics import compute_eval_metrics

#########
# Utils #
#########


class BaseSolverInterface(ABC):
    """
    Interface de base pour les solveurs de routage de câbles.
    Encapsule les différentes méthodes d'optimisation
    (Dijkstra, génétique, etc.).
    """

    def __init__(
        self,
        model_id: str,
        config: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ):
        self.config = config or {}
        self.model_id = model_id
        self.metadata = metadata or {}
        self._compute_eval_metrics = self.config.get(
            "compute_eval_metrics", True
        )

    @abstractmethod
    async def _solve_get_var_and_running_metrics(
        self, graph_data: GraphData, **kwargs
    ) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
        """
        Doit retourner le dictionnaire var et les métriques d'exécution.
        """

    async def solve(
        self, graph_data: GraphData, compute_metrics: bool = True, **kwargs
    ) -> SolverResult:
        """
        Résout le problème et construit SolverResult standardisé.

        Args:
            graph_data: Données du graphe à résoudre
            compute_metrics: Si True, calcule les métriques d'évaluation.
                Par défaut True. Peut être désactivé pour améliorer les
                performances.
            **kwargs: Arguments additionnels passés à
                _solve_get_var_and_running_metrics

        Returns:
            SolverResult avec les résultats de la résolution
        """
        start_time = time.time()
        var, runs_metrics = await self._solve_get_var_and_running_metrics(
            graph_data=graph_data, **kwargs
        )
        computation_time = time.time() - start_time

        graph_data_kwargs = graph_data.__dict__.copy()
        # Cloner df_cables pour éviter de partager les colonnes ajoutées
        if graph_data.df_cables is not None:
            graph_data_kwargs["df_cables"] = graph_data.df_cables.copy(
                deep=True
            )
        graph_data_kwargs["var"] = var
        graph_data_with_var = GraphDataWithVar(**graph_data_kwargs)

        eval_metrics: EvalMetrics | None = None
        # Le paramètre compute_metrics a priorité sur la config
        should_compute = compute_metrics and self._compute_eval_metrics
        if should_compute:
            eval_metrics = compute_eval_metrics(
                graph_data_with_var=graph_data_with_var,
                computation_time=computation_time,
                length_col_gt=self.config.get("length_col_gt", "length_gt"),
            )

        return SolverResult(
            graph_data=graph_data_with_var,
            cable_paths=var,
            eval_metrics=eval_metrics,
            model_id=self.model_id,
            runs_metrics=runs_metrics or {},
            metadata=self.metadata,
        )


if __name__ == "__main__":
    # add tests
    pass
