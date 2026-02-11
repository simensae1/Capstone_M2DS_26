import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
from scipy.spatial.distance import cdist

class RectilinearSteinerOptimizer:
    def __init__(self, gdf_nodes, source_coords):
        self.gdf = gdf_nodes.copy()
        self.source_point = Point(source_coords)
        
        # Check if source exists
        source_exists = any(p.equals(self.source_point) for p in self.gdf.geometry)
        if not source_exists:
            source_df = pd.DataFrame({'node_id': ['SOURCE'], 'geometry': [self.source_point]})
            self.gdf = pd.concat([source_df, self.gdf], ignore_index=True)
            self.gdf = gpd.GeoDataFrame(self.gdf, crs=gdf_nodes.crs)

    def _get_manhattan_path(self, p1, p2):
        """Creates a horizontal and vertical connection between two points."""
        # Corner point: (p2.x, p1.y) or (p1.x, p2.y). 
        # Using (p2.x, p1.y) creates an L-shape.
        corner = Point(p2.x, p1.y)
        if p1.x == p2.x or p1.y == p2.y:
            return LineString([p1, p2]), []
        else:
            return LineString([p1, corner, p2]), [corner]

    def _build_rectilinear_tree(self):
        coords = np.array([(p.x, p.y) for p in self.gdf.geometry])
        n = len(coords)
        
        # 1. Build MST using Manhattan distance (L1 norm)
        G = nx.Graph()
        # Manhattan distance matrix: |x1-x2| + |y1-y2|
        dist_matrix = cdist(coords, coords, metric='cityblock')
        
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=dist_matrix[i, j])
        
        mst = nx.minimum_spanning_tree(G)
        
        # 2. Convert straight MST edges to Rectilinear L-shapes
        rect_lines = []
        all_points = list(self.gdf.geometry)
        
        for u, v in mst.edges():
            p1, p2 = self.gdf.geometry.iloc[u], self.gdf.geometry.iloc[v]
            path_line, new_corners = self._get_manhattan_path(p1, p2)
            rect_lines.append(path_line)
            all_points.extend(new_corners)
            
        return unary_union(rect_lines), all_points

    def run(self):
        print(f"1. Generating Rectilinear Grid from Source ({self.source_point.x}, {self.source_point.y})...")
        multi_line, all_points = self._build_rectilinear_tree()
        print(f"   Total Manhattan Length: {multi_line.length:.2f}m")
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

optimizer = RectilinearSteinerOptimizer(gdf, source_coords=(50, 0))
result_tree, all_pts = optimizer.run()

# --- Visualization ---
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))

# Plot Rectilinear Paths
if isinstance(result_tree, MultiLineString):
    for line in result_tree.geoms:
        ax.plot(*line.xy, color='#2980b9', linewidth=3, zorder=1)
else:
    ax.plot(*result_tree.xy, color='#2980b9', linewidth=3, zorder=1)

# Node Plotting
source_p = Point(50, 0)
machine_wkt = [p.wkt for p in machine_points]

for p in all_pts:
    if p.equals(source_p):
        ax.scatter(p.x, p.y, color='gold', s=300, marker='P', zorder=5, edgecolors='black', label="Generator")
    elif p.wkt in machine_wkt:
        ax.scatter(p.x, p.y, color='#e74c3c', s=120, marker='o', zorder=5, edgecolors='black')
    else:
        # These are the corners created to force 90-degree angles
        ax.scatter(p.x, p.y, color='#95a5a6', s=40, marker='x', zorder=4, alpha=0.5)

plt.title(f"Rectilinear Electrical Grid (Manhattan Routing)\nTotal Cable: {result_tree.length:.2f}m", fontsize=14)
plt.grid(True, linestyle=':', alpha=0.6)
ax.set_aspect('equal')
plt.savefig("rectilinear_grid.png", dpi=300)
plt.show()