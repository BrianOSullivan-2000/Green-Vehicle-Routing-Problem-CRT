import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import momepy
import pandas as pd
import itertools
from shapely import geometry, ops
from shapely.geometry import Polygon


# In[1]

# Read in our dataframes
G = nx.read_gpickle("../Brians_Lab/data/dublin_graph.gpickle/dublin_graph.gpickle")

#G = ox.add_edge_speeds(G)
#G = ox.add_edge_travel_times(G)


# In[1]


# Get nodes and edges
nodes, edges, W = momepy.nx_to_gdf(G, spatial_weights=True)

nds, eds = nodes, edges

# Bounding box
lon_b = (-6.35, -6.2)
lat_b = (53.3, 53.38)

nds = nds.drop(nds[nds['x']==0].index)
nds = nds.drop(nds[nds['y']==0].index)
nds = nds.drop(nds[nds['x']>-6.2].index)
nds = nds.drop(nds[nds['x']<-6.35].index)
nds = nds.drop(nds[nds['y']>53.38].index)
nds = nds.drop(nds[nds['y']<53.3].index)

bounding_box = Polygon([[lon_b[0], lat_b[0]], [lon_b[1], lat_b[0]], [lon_b[1], lat_b[1]], [lon_b[0], lat_b[1]]])
eds = eds[eds.within(bounding_box)]


# In[1]


types = np.unique(eds.dropna(subset=['highway'])['highway'].values)
types
# Only have network of main roads
main_roads = types[[2, 3, 6, 7, 12, 13]]


eds = eds[eds['highway'].isin(main_roads)]

id_options = np.unique(edges.loc[:, 'u':'v'].values.flatten())
nds = nds[nds['osmid'].isin(id_options)]

GG = momepy.gdf_to_nx(eds, approach='primal')


# Now we have graph cleaned to just main roads


# In[1]


GGnodes = list(GG.nodes)

# Now to remove all nodes of order n=2
nds_to_remove = [n for n in GGnodes if len(list(GG.neighbors(n))) == 2]

# Need nodes and edges again
nds, eds, WW = momepy.nx_to_gdf(GG, spatial_weights=True)


# In[1]


# Make coordinates for each node
coords = []

for n in nds['geometry'].values:

    coord = n.coords.xy
    x, y = coord[0][0], coord[1][0]

    coords.append([x, y])

coords = np.array(coords)

# The index of each coordinate corresponds to the node ID
nds_to_remove = np.array(nds_to_remove)


# In[1]


# Get the node ids for all n=2 nodes
n2_ids = []
for n in nds_to_remove:

    idx = np.where(coords == n)[0][0]

    n2_ids.append(idx)

n2_ids = np.array(n2_ids)


# In[1]


# Drop duplicate rows
eds = eds.drop_duplicates(subset=['node_start', 'node_end'], keep='first')
eds = eds.loc[:, ['geometry', 'length', 'node_start', 'node_end']]

# iterative process
truecount, del_ids = 0, []
loops = 0

while (truecount < len(n2_ids)*0.95) and (loops < 200):

    n2_neighbours = np.empty((n2_ids.shape[0], 2))
    count = 0
    lines, lens, ends = [], [], []
    del_idx = []

    # Loop through each node ID
    for id in n2_ids:

        if id not in n2_neighbours.flatten():
            if id not in del_ids:

                # Find two edges and their indices
                eds2 = eds[(eds['node_start']==id) | (eds['node_end']==id)]
                idx = eds[(eds['node_start']==id) | (eds['node_end']==id)].index

                # Get the two neighbouring nodes
                nodes = np.hstack((eds2['node_start'].values, eds2['node_end'].values))
                nodes = nodes[nodes!=id]

                if len(nodes) == 2:
                    n2_neighbours[count] = nodes
                    ends.append(nodes)

                    # Combine the geometries of both edges
                    line_1, line_2 = eds2.iloc[0]['geometry'], eds2.iloc[1]['geometry']
                    newline = geometry.MultiLineString([line_1, line_2])
                    newline = ops.linemerge(newline)
                    lines.append(newline)

                    # Sum the length of both edges
                    len1, len2 = eds2.iloc[0]['length'], eds2.iloc[1]['length']
                    newlen = len1 + len2
                    lens.append(newlen)

                    del_idx.append(idx[0])
                    del_idx.append(idx[1])
                    del_ids.append(id)

                else:
                    del_ids.append(id)
                    eds.drop(idx)

                truecount += 1
        count += 1

    loops += 1
    if loops % 10 == 0:
        print(loops // 10)

    if len(lens) != 0:

        if len(lens) > 1:
            geom = np.array(lines)
        else:
            geom = lines

        new_rows = {'length':np.array(lens), 'geometry':geom, 'node_start':np.array(ends)[:, 0], 'node_end':np.array(ends)[:, 1]}
        ndf = gpd.GeoDataFrame(data=new_rows)
        eds = eds.drop(del_idx)

        eds = eds.append(ndf, ignore_index=True)


# In[1]



# Dublin shapefile (NOT IN GITHUB, go to https://www.townlands.ie/page/download/ to access)
dub_df = gpd.read_file("../Brians_Lab/data/counties.shp")
dub_df = dub_df.set_crs(epsg=4326)
dub_df = dub_df[dub_df["NAME_TAG"]=="Dublin"]


# In[1]

fig, ax = plt.subplots(1, 1, figsize=(10,10))

# Add county border
dub_df.plot(ax=ax, color="c", edgecolor="k", alpha=0.4, zorder=2)

# Plot road network and paths
graph_plot = momepy.gdf_to_nx(eds)
nds, eds = momepy.nx_to_gdf(graph_plot)

eds.plot(ax=ax, alpha=0.2, color="k", linewidth=2, zorder=3)
nds.plot(ax=ax, color='r', markersize=0.5)

# Bounds for limits
lon_b = (-6.41, -6.0843)
lat_b = (53.2294, 53.43)

# Limits for city centre
lon_b = (-6.35, -6.2)
lat_b = (53.3, 53.38)

# Plot
plt.xlim(lon_b)
plt.ylim(lat_b)
plt.show()


# In[1]


graph = momepy.gdf_to_nx(eds)

nds1, eds1 = momepy.nx_to_gdf(graph)

indices = np.arange(np.array(list(graph.nodes)).shape[0])

sample_ids = np.random.choice(indices, 100)
sample_coords = np.array(list(graph.nodes))[sample_ids]

# Create empty distance_matrix
N = len(sample_coords)
dists = np.zeros((N,N))
distance_matrix = pd.DataFrame(data=dists, index=sample_ids, columns=sample_ids)

# count for tracking
count = 0

# every node pair combination
pairs = np.array(list(itertools.combinations(sample_coords, 2)))
id_pairs = np.array(list(itertools.combinations(sample_ids, 2)))

# loop through node pairs, find path_length for each
for i in range(len(pairs)):

    pair = pairs[i]
    id_pair = id_pairs[i]

    path = np.array(nx.shortest_path(graph, tuple(pair[0]), tuple(pair[1])))

    if len([i for i in path if i in sample_coords]) == 2:

        # get path length, add to matrix, count
        path_length = nx.shortest_path_length(graph, tuple(pair[0]), tuple(pair[1]))
        distance_matrix.loc[id_pair[0], id_pair[1]] = path_length


    # impromptu progress bar
    count += 1
    if count % 1000 == 0:
        print(count // 1000)

path = np.array(nx.shortest_path(graph, tuple(pairs[0][0]), tuple(pairs[0][1])))

x = distance_matrix[distance_matrix != 0].values.flatten()
len(pairs)
len(x[~np.isnan(x)])
