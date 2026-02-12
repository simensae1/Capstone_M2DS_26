import pickle
import matplotlib.pyplot as plt
import geopandas as gpd

# 1. Load the data
file_path = "Capstone_M2DS_26/Data/graph_data_exemple.pkl"
with open(file_path, "rb") as f:
    graph_data = pickle.load(f)

# 2. Setup the plot
fig, ax = plt.subplots(figsize=(10, 10))

# 3. Plot Segments (the lines/edges)
graph_data.gdf_segments.plot(
    ax=ax, 
    color='gray', 
    linewidth=1, 
    alpha=0.7, 
    label='Segments'
)

# 4. Plot Nodes (the points/vertices)
graph_data.gdf_nodes.plot(
    ax=ax, 
    color='red', 
    markersize=5, 
    zorder=3, 
    label='Nodes'
)

# 5. Final touches
plt.title("Network Graph Visualization")
plt.xlabel("Longitude / X")
plt.ylabel("Latitude / Y")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("plot_graph_data.png", dpi=300)
plt.show()
