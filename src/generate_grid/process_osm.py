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
G = nx.read_gpickle("../Brians_Lab/data/dublin_graph.gpickle/dublin_graph.gpickle")
#G = ox.add_edge_speeds(G)
#G = ox.add_edge_travel_times(G)

# Get nodes and edges
nodes, edges, W = momepy.nx_to_gdf(G, spatial_weights=True)

# Bounding box
lon_b = (-6.41, -6.0843)
lat_b = (53.2294, 53.43)

nodes = nodes.drop(nodes[nodes['x']==0].index)
nodes = nodes.drop(nodes[nodes['y']==0].index)
nodes = nodes.drop(nodes[nodes['x']>-6.0843].index)
nodes = nodes.drop(nodes[nodes['x']<-6.41].index)
nodes = nodes.drop(nodes[nodes['y']>53.43].index)
nodes = nodes.drop(nodes[nodes['y']<53.2294].index)


# In[1]

# Clean road network data

# Speed limits and road types
speeds = ["50", "30", "80", "60", "40", "100", "120", "20"]
types = np.unique(edges['highway'].values)
types = types[[4,5,7,8,9,10,11,12,14,15]]

# Pandas filtering
edges = edges[edges['maxspeed'].isin(speeds)]
edges = edges[edges['highway'].isin(types)]


# In[2]

# Get sample nodes for nodes
ids = nodes["osmid"].values
sample_ids = np.random.choice(ids, 20)

# Create empty distance_matrix
N = len(sample_ids)
dists = np.zeros((N,N))
distance_matrix = pd.DataFrame(data=dists, index=sample_ids, columns=sample_ids)

# count for tracking
count = 0

# every node pair combination
pairs = np.array(list(itertools.combinations(sample_ids, 2)))


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


# In[1]

# Save distance matrix
distance_matrix.to_json("data/distance_matrices/n20.json")
