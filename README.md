# Capstone_M2DS_26

Version light d'étude de solver et de sampling pour le routage de câbles.

## Description

Ce projet est une version simplifiée et autonome pour l'étude de solveurs de routage de câbles. Il contient les composants essentiels pour :
- La gestion des données de graphe (GraphData)
- L'implémentation de solveurs de plus court chemin (ShortestPathSolver)
- Le calcul de métriques d'évaluation
- La construction de différents types de graphes (NetworkX, IGraph, graphe de segments)

## Structure du projet

```
Capstone_M2DS_26/
├── Data/                          # Données et cache
│   └── graph_data_cache.pkl       # Cache des données de graphe
├── dataclass/                     # Classes de données
│   ├── graph_data.py              # GraphData : format pivot pour les données de graphe
│   ├── graph_data_with_var.py     # GraphDataWithVar : extension avec variables de décision
│   └── solver_results.py          # SolverResult et EvalMetrics : résultats standardisés
├── solver/                        # Solveurs
│   ├── base_solver.py             # BaseSolverInterface : interface de base pour les solveurs
│   ├── shortest_path_solver.py    # ShortestPathSolver : solveur de plus court chemin
│   ├── hgraph/                    # Graphes hybrides
│   │   ├── base_hgraph.py        # BaseHGraph : interface de base pour les graphes
│   │   └── hgraph_simple.py       # GraphSimple : wrapper pour NetworkX/Igraph
│   └── utils/                     # Utilitaires
│       ├── compute_metrics.py    # Calcul des métriques d'évaluation
│       ├── cost_function.py      # Fonctions de coût pour l'optimisation
│       └── graph_builders.py      # Construction de graphes (NetworkX, IGraph, segments)
└── README.md                       # Documentation du projet
```

## Classes principales

### GraphData (`dataclass/graph_data.py`)
Format pivot utilisant GeoPandas pour unifier géométries et relations. Contient :
- `gdf_segments` : GeoDataFrame avec géométries LineString
- `gdf_nodes` : GeoDataFrame avec géométries Point
- `nodes_coordinates` : Array numpy des coordonnées des nœuds
- `df_cables` : DataFrame avec informations des câbles
- `df_angles` : DataFrame avec informations d'angles
- `m_per_unit` : Facteur de conversion des unités

### GraphDataWithVar (`dataclass/graph_data_with_var.py`)
Extension de GraphData qui embarque aussi les chemins résolus (`var`).

### SolverResult (`dataclass/solver_results.py`)
Format pivot pour standardiser la sortie des solveurs. Contient :
- `graph_data` : Données du graphe avec variables
- `cable_paths` : Dictionnaire des chemins résolus
- `eval_metrics` : Métriques d'évaluation standardisées
- `runs_metrics` : Métriques spécifiques au modèle
- `model_id` : Identifiant du modèle/solveur

### EvalMetrics (`dataclass/solver_results.py`)
Métriques d'évaluation standardisées pour tous les solveurs (temps de calcul, couverture, longueurs, poids, angles, etc.).

### BaseSolverInterface (`solver/base_solver.py`)
Interface de base pour les solveurs de routage de câbles. Encapsule les différentes méthodes d'optimisation.

### ShortestPathSolver (`solver/shortest_path_solver.py`)
Solveur de plus court chemin utilisant Dijkstra ou A*. Supporte différents types de graphes :
- NetworkX standard
- IGraph
- NetworkX segments (graphe où les nœuds sont des segments)

### BaseHGraph (`solver/hgraph/base_hgraph.py`)
Interface de base pour la gestion des graphes. Encapsule les opérations minimales sur le graphe.

### GraphSimple (`solver/hgraph/hgraph_simple.py`)
Wrapper unique pour NetworkX ou Igraph, piloté par la config `graph_type`. Stocke les données nécessaires au calcul des métriques.

### compute_metrics (`solver/utils/compute_metrics.py`)
Fonctions pour calculer les métriques d'exécution et d'évaluation :
- `shortest_path_running_metrics` : Métriques pendant l'exécution
- `compute_eval_metrics` : Métriques d'évaluation finales

### cost_function (`solver/utils/cost_function.py`)
Fonctions pour calculer les fonctions de coût à partir d'un hgraph.

### graph_builders (`solver/utils/graph_builders.py`)
Fonctions pour construire différents types de graphes :
- `build_networkx_graph` : Construction d'un graphe NetworkX
- `build_igraph_graph` : Construction d'un graphe IGraph
- `build_networkx_segment_graph` : Construction d'un graphe de segments

## Utilisation

### Exemple basique

```python
from dataclass.graph_data import GraphData
from solver.shortest_path_solver import ShortestPathSolver
import asyncio

# Charger les données (depuis cache ou autre source)
graph_data = load_graph_data()

# Créer le solveur
solver = ShortestPathSolver(
    config={
        "graph_type": "networkx",
        "solve_mode": "dijkstra",
        "edge_weight_attr": "length_m",
    }
)

# Résoudre
solver_result = asyncio.run(
    solver.solve(graph_data=graph_data, compute_metrics=True)
)

# Accéder aux résultats
print(f"Couverture: {solver_result.eval_metrics.coverage_pct:.2f}%")
print(f"Temps: {solver_result.eval_metrics.computation_time:.2f} s")
```

## Installation

### Prérequis

- Python 3.8 ou supérieur

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## Dépendances

Les dépendances principales sont listées dans `requirements.txt` :
- numpy : Calculs numériques
- pandas : Manipulation de données
- geopandas : Données géospatiales
- networkx : Manipulation de graphes
- python-igraph : Bibliothèque de graphes alternative
- shapely : Opérations géométriques
- pyproj : Projections cartographiques
- tqdm : Barres de progression

## Notes

Ce projet est une version autonome et simplifiée. Les données doivent être chargées depuis un cache ou une autre source externe.
