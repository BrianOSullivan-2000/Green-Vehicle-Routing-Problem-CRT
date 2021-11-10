import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import momepy
import pandas as pd
import itertools


# In[1]

# Read in our dataframes
G = nx.read_gpickle("data/dublin_graph.gpickle")
#G = ox.add_edge_speeds(G)
#G = ox.add_edge_travel_times(G)

# Get nodes and edges
nodes, edges, W = momepy.nx_to_gdf(G, spatial_weights=True)


# In[2]

# Get sample ids for nodes
ids = nodes["osmid"].values
sample_ids = np.random.choice(ids, 50)

# Create empty distance_matrix
N = len(sample_ids)
dists = np.zeros((N,N))
distance_matrix = pd.DataFrame(data=dists, index=sample_ids, columns=sample_ids)

# count for tracking
count = 0

# every node pair combination
pairs = np.array(list(itertools.combinations(sample_ids, 2)))


# In[1]

# Save file
pd.read_json("data/distance_matrices/n50.json")


# In[1]

# loop through node pairs, find path_length for each
for pair in pairs:

    # get path length, add to matrix, count
    path_length = nx.shortest_path_length(G, pair[0], pair[1], weight="length")
    distance_matrix.loc[pair[0], pair[1]] = path_length
    count += 1

    # impromptu progress bar
    if count % 50 == 0:
        print(count // 50)


distance_matrix.to_csv("data/distance_matrices/n50.csv")
