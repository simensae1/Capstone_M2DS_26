# -*- coding: utf-8 -*-
############
# Packages #
############
import sys
from pathlib import Path
from typing import Any, Dict
from abc import ABC, abstractmethod

sys.path.append(str(Path(__file__).parents[2]))
#######################
# Internal Imports #
#######################

#########
# Utils #
#########


class BaseHGraph(ABC):
    """
    Interface de base pour la gestion des graphes.
    Encapsule les opérations minimales sur le graphe.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.graph = None

    @abstractmethod
    def get_var(self) -> Dict[str, Any]:
        """
        Retourne les variables à optimiser.

        Returns:
            Dictionnaire de variables
        """
        pass

    @abstractmethod
    def update_var(self, var: Dict[str, Any]) -> None:
        """
        Met à jour les variables à optimiser.

        Args:
            var: Dictionnaire de variables
        """
        pass

    @abstractmethod
    def get_subgraph(self, filters: Dict[str, Any]) -> Any:
        """
        Retourne un sous-graphe selon des filtres.

        Args:
            filters: Dictionnaire de filtres

        Returns:
            Sous-graphe NetworkX ou autre type de graphe
        """
        pass


if __name__ == "__main__":
    # add tests
    pass
