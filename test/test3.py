import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
from scipy.spatial.distance import cdist

class SteinerTreeOptimizer:
    def __init__(self, gdf_nodes, source_coords, angle_penalty):
        self.gdf = gdf_nodes.copy()
        self.source_point = Point(source_coords)
        self.penalty = angle_penalty
        
        # Check if source exists using .equals()
        source_exists = any(p.equals(self.source_point) for p in self.gdf.geometry)
        
        if not source_exists:
            source_df = pd.DataFrame({'node_id': ['SOURCE'], 'geometry': [self.source_point]})
            self.gdf = pd.concat([source_df, self.gdf], ignore_index=True)
            self.gdf = gpd.GeoDataFrame(self.gdf, crs=gdf_nodes.crs)

    def _get_steiner_point(self, p1, p2, p3):
        """Calculates Fermat point heuristic for smoother branching."""
        return Point((p1.x + p2.x + p3.x) / 3, (p1.y + p2.y + p3.y) / 3)

    def _build_tree(self):
        coords = np.array([(p.x, p.y) for p in self.gdf.geometry])
        n = len(coords)
        
        # 1. Initial MST
        G = nx.Graph()
        dist_matrix = cdist(coords, coords)
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=dist_matrix[i, j])
        mst = nx.minimum_spanning_tree(G)
        
        # 2. Add Steiner Points for junctions with > 2 connections
        steiner_coords = []
        for node in list(mst.nodes()):
            neighbors = list(mst.neighbors(node))
            if len(neighbors) >= 3:
                p_center = self.gdf.geometry.iloc[node]
                p_n1 = self.gdf.geometry.iloc[neighbors[0]]
                p_n2 = self.gdf.geometry.iloc[neighbors[1]]
                steiner_coords.append(self._get_steiner_point(p_center, p_n1, p_n2))
        
        # 3. Final MST including the new Steiner points
        all_points = list(self.gdf.geometry) + steiner_coords
        all_coords = np.array([(p.x, p.y) for p in all_points])
        
        final_G = nx.Graph()
        final_dist = cdist(all_coords, all_coords)
        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                final_G.add_edge(i, j, weight=final_dist[i, j])
        
        return nx.minimum_spanning_tree(final_G), all_points

    def run(self):
        print(f"1. Optimizing network with Source at ({self.source_point.x}, {self.source_point.y})...")
        tree, all_points = self._build_tree()
        
        lines = []
        for u, v in tree.edges():
            lines.append(LineString([all_points[u], all_points[v]]))
        
        multi_line = unary_union(lines)
        print(f"   Total Cable Length: {multi_line.length:.2f}m")
        return multi_line, all_points

# --- Execution ---
machine_points = [
    Point(0, 100), Point(0, 50), Point(10, 10), 
    Point(100, 0), Point(80, 20), Point(45, 70), Point(10, 30), Point(90,90), Point(80,10)
]

gdf = gpd.GeoDataFrame({
    'node_id': [f"M{i}" for i in range(len(machine_points))], 
    'geometry': machine_points
}, crs="EPSG:2154")

optimizer = SteinerTreeOptimizer(gdf, source_coords=(50, 0), angle_penalty=0)
result_tree, all_pts = optimizer.run()

# --- Visualization ---
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))

# Path plotting
if isinstance(result_tree, MultiLineString):
    for line in result_tree.geoms:
        ax.plot(*line.xy, color='#34495e', linewidth=2.5, zorder=1)
else:
    ax.plot(*result_tree.xy, color='#34495e', linewidth=2.5, zorder=1)

# Node plotting logic using .equals()
source_p = Point(50, 0)
machine_wkt = [p.wkt for p in machine_points]

for p in all_pts:
    if p.equals(source_p):
        color, label, size, marker = 'gold', 'Generator', 250, 'P'
    elif p.wkt in machine_wkt:
        color, label, size, marker = '#e74c3c', 'Machine', 120, 'o'
    else:
        color, label, size, marker = '#2ecc71', 'Junction', 70, 's'
    
    ax.scatter(p.x, p.y, color=color, s=size, marker=marker, zorder=5, edgecolors='black')

plt.title(f"Electrical Steiner Tree Optimization\nCable Length: {result_tree.length:.2f}m", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
ax.set_aspect('equal')
plt.savefig("final_electrical_grid.png", dpi=300)
plt.show()
