# -*- coding: utf-8 -*-
############
# Packages #
############
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parents[1]))
#######################
# Internal Imports #
#######################
from dataclass.graph_data import GraphData

#########
# Utils #
#########


@dataclass
class GraphDataWithVar(GraphData):
    """
    Extension légère de GraphData qui embarque aussi les chemins résolus.
    """
    var: Dict[int, List[int]] = field(default_factory=dict)

    def update_var(self, var_data: Dict[int, List[int]]) -> None:
        """
        Met à jour les chemins de câbles (remplacement ou ajout).
        """
        self.var.update(var_data)


if __name__ == "__main__":
    # add tests
    pass
