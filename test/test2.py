import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge, unary_union

class TreeOptimizer:
    def __init__(self, gdf_nodes, angle_penalty):
        self.gdf = gdf_nodes.copy()
        self.penalty = angle_penalty

    def _build_steiner_tree(self):
        """Creates a branching tree connecting all nodes."""
        from scipy.spatial.distance import pdist, squareform
        
        coords = np.array([(p.x, p.y) for p in self.gdf.geometry])
        dist_matrix = squareform(pdist(coords))
        
        # Create a complete graph of all possible connections
        G = nx.Graph()
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                G.add_edge(i, j, weight=dist_matrix[i, j])
        
        # In a complete graph, the Steiner Tree for all nodes 
        # is essentially the Minimum Spanning Tree (MST).
        # This allows the path to 'split' at any node.
        mst = nx.minimum_spanning_tree(G)
        return mst

    def run(self):
        print("1. Calculating optimal branching structure (MST)...")
        mst = self._build_steiner_tree()
        
        # Convert MST edges to a list of lines
        lines = []
        for u, v in mst.edges():
            line = LineString([self.gdf.geometry.iloc[u], self.gdf.geometry.iloc[v]])
            lines.append(line)
        
        # Combine all lines into a single geometric object
        multi_line = unary_union(lines)
        
        # Metrics Calculation
        total_length = multi_line.length
        # In an MST, number of edges is always nodes - 1
        num_edges = mst.number_of_edges()
        
        print(f"   Final Total Length: {total_length:.2f}m")
        print(f"   Number of segments (edges): {num_edges}")
        print(f"   Number of connection points: {len(self.gdf)}")
        
        return multi_line

# --- Execution ---
points = [
        Point(0, 100),  # Top
        Point(0, 50),   # Middle Vertical
        Point(10, 10),  # Near Corner
        Point(50, 0),   # Middle Horizontal
        Point(100, 0),   # Right
        Point(80, 20),
        Point(45, 70),
        Point(10, 30)
]

gdf = gpd.GeoDataFrame({
    'node_id': range(len(points)),
    'geometry': points
}, crs="EPSG:2154")

optimizer = TreeOptimizer(gdf, angle_penalty=50)
result_tree = optimizer.run()

# Visualization
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the branching tree
if isinstance(result_tree, MultiLineString):
    for line in result_tree.geoms:
        x, y = line.xy
        ax.plot(x, y, color='blue', linewidth=3, alpha=0.6, label='Tree Edge' if line == result_tree.geoms[0] else "")
else:
    x, y = result_tree.xy
    ax.plot(x, y, color='blue', linewidth=3, alpha=0.6, label='Tree Edge')

gdf.plot(ax=ax, color='red', markersize=100, zorder=5, label="Nodes")

# Optional: adding text labels for node IDs to verify the branching
for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.node_id):
    ax.text(x + 2, y + 2, label, fontsize=12)

plt.title(f"Branching Path (MST)\nTotal Length: {result_tree.length:.2f}m")
plt.legend()
plt.grid(True)
plt.savefig("optimized_path2.png", dpi=300, bbox_inches='tight')
print("Image saved as optimized_path2.png")
plt.show()