import geopandas as gpd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


class PathOptimizer:
    def __init__(self, gdf_nodes, angle_penalty_meters):
        """
        :param gdf_nodes: GeoDataFrame with 'node_id' and geometry (Point)
        :param angle_penalty_meters: The 'virtual distance' cost of making a turn.
                                     High value = fewer turns (more straight lines).
                                     Low value = shortest physical path.
        """
        self.gdf = gdf_nodes.copy()
        self.angle_penalty = angle_penalty_meters

        # Ensure we are working with a projected CRS (meters), not Lat/Lon
        if self.gdf.crs.is_geographic:
            raise ValueError("GeoDataFrame must be in a projected CRS (e.g., EPSG:2154 for France) to calculate distances correctly.")

    def _solve_open_tsp(self):
        coords = np.array([(p.x, p.y) for p in self.gdf.geometry])
        n = len(coords)

        dist_matrix = squareform(pdist(coords))
        G = nx.from_numpy_array(dist_matrix)

        # Add Dummy Node (Index n)
        G.add_node(n)
        for i in range(n):
            G.add_edge(n, i, weight=0)

        # Solve TSP
        cycle = nx.approximation.traveling_salesman_problem(G, cycle=True)

        # 1. Find where the dummy node (n) is in the cycle
        dummy_pos = cycle.index(n)

        # 2. Re-order the cycle so it starts/ends at the dummy node
        # This makes the path 'open'
        reordered = cycle[dummy_pos:] + cycle[:dummy_pos]

        # 3. CRITICAL FIX: Remove the dummy node index 'n'
        # so we only have valid indices for gdf.iloc
        path_indices = [i for i in reordered if i != n]

        # Now iloc will not go out of bounds
        ordered_points = [self.gdf.geometry.iloc[i] for i in path_indices]
        return ordered_points

    def _get_line_intersection(self, A, B, C, D):
        """
        Finds intersection S of Line(A,B) and Line(C,D).
        Returns S (Point) if it exists and is geometrically valid for extension, else None.
        """
        # Line AB represented as a1x + b1y = c1
        a1 = B.y - A.y
        b1 = A.x - B.x
        c1 = a1 * A.x + b1 * A.y

        # Line CD represented as a2x + b2y = c2
        a2 = D.y - C.y
        b2 = C.x - D.x
        c2 = a2 * C.x + b2 * C.y

        determinant = a1 * b2 - a2 * b1

        if abs(determinant) < 1e-6:
            return None  # Parallel lines

        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        S = Point(x, y)

        # VALIDATION LOGIC:
        # We are looking to replace A->B->C->D with A->B->S->C->D
        # This implies S must be "ahead" of B (relative to A)
        # and S must be "behind" C (relative to D, effectively ahead of C in reverse).

        # Check 1: S is not "behind" B. Dot product check.
        # Vector BS dot Vector AB must be positive
        vec_AB = np.array([B.x - A.x, B.y - A.y])
        vec_BS = np.array([S.x - B.x, S.y - B.y])

        # Check 2: S is not "behind" C.
        # Vector CS dot Vector DC must be positive
        vec_DC = np.array([C.x - D.x, C.y - D.y])
        vec_CS = np.array([S.x - C.x, S.y - C.y])

        if np.dot(vec_AB, vec_BS) > 0 and np.dot(vec_DC, vec_CS) > 0:
            return S

        return None

    def _optimize_geometry(self, path):
        """
        Iteratively finds intersections to reduce angles.
        """
        points = list(path)  # List of Shapely Points
        improved = True

        while improved:
            improved = False
            i = 0
            while i < len(points) - 3:
                A = points[i]
                B = points[i+1]
                C = points[i+2]
                D = points[i+3]

                # Try to find an intersection 'S' extending AB and DC
                S = self._get_line_intersection(A, B, C, D)

                if S:
                    # Current Cost: 3 segments + 2 turns (at B and C)
                    current_len = A.distance(B) + B.distance(C) + C.distance(D)
                    current_cost = current_len + (2 * self.angle_penalty)

                    # New Cost: 2 long segments + 1 turn (at S)
                    # Path becomes A -> B -> S -> C -> D
                    # Note: B and C are collinear with S, so B and C are no longer "turns"
                    new_len = A.distance(S) + S.distance(D)  # (A->B->S) + (S->C->D)
                    new_cost = new_len + (1 * self.angle_penalty)

                    if new_cost < current_cost:
                        # Apply Optimization: Insert S
                        # We keep B and C in the list to maintain connectivity,
                        # but they are now straight pass-throughs.
                        points.insert(i+2, S)
                        improved = True
                        # Skip ahead to avoid re-optimizing the immediate vicinity endlessly
                        i += 2
                i += 1
        return points

    def run(self):
        print("1. Calculating optimal visiting order (TSP)...")
        initial_path = self._solve_open_tsp()

        print(f"   Initial Length: {LineString(initial_path).length:.2f}m")
        print("2. Optimizing geometry (reducing turns)...")
        final_points = self._optimize_geometry(initial_path)

        final_line = LineString(final_points)
        print(f"   Final Length: {final_line.length:.2f}m")
        print(f"   Nodes count: {len(initial_path)} -> {len(final_points)} (Added {len(final_points) - len(initial_path)} Steiner points)")

        return final_line

# ==========================================
# EXAMPLE USAGE
# ==========================================

# 1. Create Dummy Data (e.g., L-Shape points)


data = {
    'node_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'geometry': [
        Point(0, 100),  # Top
        Point(0, 50),   # Middle Vertical
        Point(10, 10),  # Near Corner
        Point(50, 0),   # Middle Horizontal
        Point(100, 0),   # Right
        Point(80, 20),
        Point(45, 70),
        Point(10, 30)
    ]
}
gdf = gpd.GeoDataFrame(data, crs="EPSG:2154")  # Lambert 93 (France) or generic meters

# 2. Run Algorithm
# penalty=50 means: "I am willing to travel 50m extra to avoid 1 turn"
optimizer = PathOptimizer(gdf, angle_penalty_meters=500)
result_line = optimizer.run()

# 3. Visualize
fig, ax = plt.subplots(figsize=(8, 8))
gpd.GeoSeries([result_line]).plot(ax=ax, color='blue', alpha=0.5, linewidth=3, label='Optimized Path')
gdf.plot(ax=ax, color='red', marker='o', markersize=50, label='Original Nodes', zorder=5)

# Plot the vertices of the result line to see Steiner points
x, y = result_line.xy
ax.scatter(x, y, color='green', marker='x', s=30, label='Path Vertices (inc. Steiner)', zorder=6)

plt.legend()
plt.title("Path Optimization: Min Length + Min Angles")
plt.grid(True)
plt.savefig("optimized_path.png", dpi=300, bbox_inches='tight')
plt.show()
