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

G = ox.add_edge_speeds(G)
#G = ox.add_edge_travel_times(G)


# In[1]


# Get nodes and edges
nodes, edges, W = momepy.nx_to_gdf(G, spatial_weights=True)

nds, eds = nodes, edges

# M50 bounding box
#lon_b = (-6.391, -6.040)
#lat_b = (53.227, 53.414)

# City Centre Bounding box
lon_b = (-6.32, -6.21)
lat_b = (53.325, 53.365)

# South Dublin Bounding box
#lon_b = (-6.375, -6.081)
#lat_b = (53.230, 53.325)


# Drop nodes outside bounding box
nds = nds.drop(nds[nds['x']==0].index)
nds = nds.drop(nds[nds['y']==0].index)
nds = nds.drop(nds[nds['x']>lon_b[1]].index)
nds = nds.drop(nds[nds['x']<lon_b[0]].index)
nds = nds.drop(nds[nds['y']>lat_b[1]].index)
nds = nds.drop(nds[nds['y']<lat_b[0]].index)

# Drop edges outside bounding box
bounding_box = Polygon([[lon_b[0], lat_b[0]], [lon_b[1], lat_b[0]], [lon_b[1], lat_b[1]], [lon_b[0], lat_b[1]]])
eds = eds[eds.within(bounding_box)]


# In[1]


# Get all OSM road types
types = np.unique(eds.dropna(subset=['highway'])['highway'].values)

# check the indices
# We want to generally go with motorway, trunk, primary, secondary, and all the corresponding link roads
types

# Only have network of main roads

# M50 indices
# main_roads = types[[3, 4, 6, 7, 9, 10, 12, 13, 15, 16]]

# City Centre indices
main_roads = types[[2, 3, 4, 5, 6, 8, 9, 10, 11]]

# South Dublin indices
# main_roads = types[[3, 4, 6, 7, 9, 10, 12, 13, 15, 16]]

eds = eds[eds['highway'].isin(main_roads)]


GG = momepy.gdf_to_nx(eds, approach='primal')

# Don't need multigraph, just graph
GG = nx.Graph(GG)

# Now we have graph cleaned to just main roads

# Dublin shapefile (NOT IN GITHUB, go to https://www.townlands.ie/page/download/ to access)
dub_df = gpd.read_file("../Brians_Lab/data/counties.shp")
dub_df = dub_df.set_crs(epsg=4326)
dub_df = dub_df[dub_df["NAME_TAG"]=="Dublin"]

# Remove any small disconnected elements

while nx.number_connected_components(GG) > 1:
    for component in list(nx.connected_components(GG)):
        if len(component) <= min([len(l) for l in list(nx.connected_components(GG))]):
            for node in component:
                print("node")
                GG.remove_node(node)

# Should be 1
nx.number_connected_components(GG)


# In[1]


if nx.number_connected_components(GG) > 1:
    for component in list(nx.connected_components(GG)):
        if len(component) <= min([len(l) for l in list(nx.connected_components(GG))]):
            for node in component:
                GG.remove_node(node)

# List of nodes
GGnodes = list(GG.nodes)

# Now to remove all nodes of order n=2
nds_to_remove = [n for n in GGnodes if len(list(GG.neighbors(n))) == 2]

# Need nodes and edges again (I do this a few times, it just makes the code play nice)
nds, eds, WW = momepy.nx_to_gdf(GG, spatial_weights=True)

# Make coordinates for each node
coords = []

# Loop through geometry of each node and extract lon and lat
for n in nds['geometry'].values:

    coord = n.coords.xy
    x, y = coord[0][0], coord[1][0]

    coords.append([x, y])

coords = np.array(coords)

# The index of each coordinate corresponds to the node ID
nds_to_remove = np.array(nds_to_remove)

# Get the node ids for all n=2 nodes
n2_ids = []
for n in nds_to_remove:

    idx = np.where(coords == n)[0][0]
    n2_ids.append(idx)

n2_ids = np.array(n2_ids)

# Drop duplicate rows and clean dataframe to relevant columns

eds = eds.drop_duplicates(subset=['node_start', 'node_end'], keep='first')

eds = eds.loc[:, ['highway', 'maxspeed', 'geometry', 'length', 'node_start', 'node_end']]

eds = eds[~eds['maxspeed'].isnull().values]
eds['maxspeed'] = eds['maxspeed'].values.astype(float)

def most_frequent(List):

    # Return most elements of list, only do it if the element is a list though
    if isinstance(List, list):
        return max(set(List), key = List.count)
    else:
        return List


# iterative process to remove each n=2 node (have to run this chunk a few times)
# del_ids are node ids that have already been deleted
truecount, del_ids = 0, []
loops = 0


id = n2_ids[0]

while (truecount < len(n2_ids)) and (loops < 10):

    # various elements such as lengths, shapes and node ends for all new edges
    # are first save in lists, then added as pandas dataframe rows at the end
    n2_neighbours = np.empty((n2_ids.shape[0], 2))
    count = 0
    highways, lines, lens, ends = [], [], [], []
    del_idx = []

    # Loop through each node ID
    for id in n2_ids:

        # n2_neighbours list ensures that neighbouring n=2 nodes are not both processed in the same loop
        # the del_ids list simply ensures only nodes that still exist are deleted
        if id not in n2_neighbours.flatten():
            if id not in del_ids:

                # Find two edges and their indices
                eds2 = eds[(eds['node_start']==id) | (eds['node_end']==id)]
                idx = eds[(eds['node_start']==id) | (eds['node_end']==id)].index

                # Get the two neighbouring nodes
                nodes = np.hstack((eds2['node_start'].values, eds2['node_end'].values))
                nodes = nodes[nodes!=id]

                # two neighbouring nodes, great, time to make an edge between them
                if len(nodes) == 2:

                    # record end nodes
                    n2_neighbours[count] = nodes
                    ends.append(nodes)

                    # Get all the highway elements of the two neighbouring edges,
                    # record them in the new combined edge
                    highway = []
                    for h in eds2['highway'].values:
                        if isinstance(h, str):
                            highway.append(h)
                        else:
                            for hh in h:
                                highway.append(hh)
                    highways.append(highway)

                    # Combine the geometries of both edges
                    line_1, line_2 = eds2.iloc[0]['geometry'], eds2.iloc[1]['geometry']
                    newline = geometry.MultiLineString([line_1, line_2])
                    newline = ops.linemerge(newline)
                    lines.append(newline)


                    # Sum the length of both edges
                    len1, len2 = eds2.iloc[0]['length'], eds2.iloc[1]['length']
                    newlen = len1 + len2
                    lens.append(newlen)

                    # Find average speed limit
                    speed = (float(eds2.iloc[0]['maxspeed'])*(len1/newlen) + float(eds2.iloc[1]['maxspeed'])*(len2/newlen))

                    # del_idx are the indices of rows in eds that need to be removed
                    del_idx.append(idx[0])
                    del_idx.append(idx[1])
                    del_ids.append(id)

                else:
                    # Some nodes that are not n=2, don't know why they make it this far but around 4 do
                    del_ids.append(id)
                    eds.drop(idx)

                truecount += 1
        count += 1

    # safety measure to prevent hanging
    loops += 1

    # progress tracker
    # if loops % 10 == 0:
        # print(loops // 10)

    # only add edge to dataframe if it exists
    if len(lens) != 0:

        # lines can be awkward if wrong number
        if len(lens) > 1:
            geom = np.array(lines)
        else:
            geom = lines

        # get data for new row and add to dataframe, drop old edges
        new_rows = {'highway': highways, 'maxspeed':np.array(speed), 'length':np.array(lens), 'geometry':geom, 'node_start':np.array(ends)[:, 0], 'node_end':np.array(ends)[:, 1]}
        ndf = gpd.GeoDataFrame(data=new_rows)
        eds = eds.drop(del_idx)

        eds = eds.append(ndf, ignore_index=True)

        # Our new edges have multiple highway entries, get the most common one and assign it
        eds["highway"] = eds['highway'].apply(most_frequent)


# Another graph/gdf reset
GG = momepy.gdf_to_nx(eds)
nds, eds = momepy.nx_to_gdf(GG)


# Let's plot the results
fig, ax = plt.subplots(1, 1, figsize=(10,10))

# Add county border
dub_df.plot(ax=ax, color="c", edgecolor="k", alpha=0.4, zorder=2)

# Plot edges and nodes
eds.plot(ax=ax, alpha=0.2, color="k", linewidth=2, zorder=3)
nds.plot(ax=ax, color='crimson', markersize=5)

# Bounds to zoom in on a small cluster (O'Connell Bridge)
#lon_b = (-6.265, -6.25)
#lat_b = (53.343, 53.35)

# Plot
plt.xlim(lon_b)
plt.ylim(lat_b)

# plt.savefig("data/figures/dublin_centre_k2.jpeg", dpi=300)
# nx.write_gpickle(GG, "data/subgraphs/dublin_south_junctions.gpickle")
print(eds.shape)
plt.show()


# In[1]


# Code for contracting edges, simplifies node clusters in graph

GG = nx.read_gpickle("data/subgraphs/dublin_south_junctions.gpickle")

# Make new graph to iterate over
con_graph = GG.copy()
nds, eds = momepy.nx_to_gdf(GG)
geom_err = 0
count = 0


# Remove shortest edges
for i in range(500):

    # Get smallest edge and get node coordinates
    edge = eds[eds['length'] == np.min(eds['length'].values)]
    nodes = list(con_graph.edges)[edge.index[0]]

    # Identify nodeIDs of edge to be removed (first node is always kept)
    points = [geometry.Point(nodes[0]), geometry.Point(nodes[1])]
    nodeIDs = list(nds[nds['geometry'].isin(points)]['nodeID'])

    # Find all edges connected to node to be dropped
    drop_eds = eds[(eds['node_start']==nodeIDs[1]) | (eds['node_end']==nodeIDs[1])]

    # Find all edges connected to node not to be dropped and corresponding ids
    alt_eds = eds[(eds['node_start']==nodeIDs[0]) | (eds['node_end']==nodeIDs[0])]
    alt_ids = alt_eds.loc[:, 'node_start':'node_end'].values.flatten()
    alt_ids = alt_ids[alt_ids != nodeIDs[0]]


    # Get the node IDs that are connected to node that will be deleted, but are
    # not already connected to the node that will be kept

    new_ids = drop_eds.loc[drop_eds['length']!=drop_eds['length'].min()].loc[:, 'node_start':'node_end'].values.flatten()
    new_ids = new_ids[new_ids != nodeIDs[1]]
    id_keep = ~np.isin(new_ids, alt_ids)
    new_ids = new_ids[id_keep]

    # The IDs of nodes which are greater than the node to be dropped are reduced by one
    new_ids[new_ids>nodeIDs[1]] = new_ids[new_ids>nodeIDs[1]] - 1

    # Adjust length of all these edges by adding the length of edge to be dropped
    # Also adjust linestrings in similar fashion
    len_list = list(drop_eds['length'])
    geom_list = list(drop_eds['geometry'])
    speed_list = list(drop_eds['maxspeed'])
    highway_list = list(drop_eds['highway'])

    # edge index to be dropped
    drop_index = len_list.index(min(len_list))

    # Pop out length and geometry of dropped edge
    add_len = len_list.pop(drop_index)
    add_geom = geom_list.pop(drop_index)
    add_speed = speed_list.pop(drop_index)
    drop_highway = highway_list.pop(drop_index)

    # Add this to kept edges
    len_list = np.array(len_list)[id_keep] + add_len

    # Create new geometries to account for removed node
    geom_list = [geom_list[g] for g in list(np.arange((len(geom_list)))[id_keep])]

    # Add removed edge geometry to all kept edges
    for g in range(len(geom_list)):

        geom = geom_list[g]
        newline = geometry.MultiLineString([geom, add_geom])
        newline = ops.linemerge(newline)
        geom_list[g] = newline

    # Adjust speed accordingly
    speed_list = np.array(speed_list).astype(float)[id_keep] * ((len_list-add_len)/len_list)
    speed_list = speed_list + ((add_speed * add_len) / len_list)

    # Contract the two nodes (The main step)
    con_graph = nx.contracted_nodes(con_graph, nodes[0], nodes[1], self_loops=False)
    nds, eds = momepy.nx_to_gdf(con_graph)


    # Drop duplicates edges which arise from any triangles
    eds = eds.drop_duplicates(subset=["node_start", "node_end"])

    # Give all newly generated edges their true length (the sum of removed edge and previous edge) and true geometry
    eds.loc[((eds['node_start']==nodeIDs[0]) & (eds['node_end'].isin(new_ids)))
            | ((eds['node_start'].isin(new_ids)) & (eds['node_end']==nodeIDs[0])), 'length'] = len_list

    # Do the same for speed limit
    eds.loc[((eds['node_start']==nodeIDs[0]) & (eds['node_end'].isin(new_ids)))
            | ((eds['node_start'].isin(new_ids)) & (eds['node_end']==nodeIDs[0])), 'maxspeed'] = speed_list

    # Similarly, give all newly generated edges their true geometries, about 6 errors from disjointed edges
    g_idx = 0
    for idx in eds.loc[((eds['node_start']==nodeIDs[0]) & (eds['node_end'].isin(new_ids)))
                       | ((eds['node_start'].isin(new_ids)) & (eds['node_end']==nodeIDs[0])), 'geometry'].index:

        try:
            final_line = geom_list[g_idx]
            eds.loc[idx, 'geometry'] = final_line

        except:
            geom_err += 1

        finally:
            g_idx += 1



    # impromptu progress bar
    count += 1
    if count % 100 == 0:
        print(count // 100)

    # Another reset to prepare for next iteration
    con_graph = momepy.gdf_to_nx(eds)
    nds, eds = momepy.nx_to_gdf(con_graph)

print(nds.shape)


# nx.write_gpickle(con_graph, "data/subgraphs/dublin_south_clean.gpickle")


# In[1]


# Plot the newly cleaned graph
fig, ax = plt.subplots(1, 1, figsize=(10,10))

# Add county border
dub_df.plot(ax=ax, color="c", edgecolor="k", alpha=0.4, zorder=2)

# Plot edges and nodes
eds.plot(ax=ax, alpha=1, color="silver", linewidth=2, zorder=1)
nds.plot(ax=ax, color='crimson', markersize=7)


# Dublin Circular North and South
lon_b = (-6.4, -6.07)
lat_b = (53.2, 53.33)

# O'Connell Bridge
#lon_b = (-6.265, -6.25)
#lat_b = (53.343, 53.35)
#lon_b = (-6.389, -6.385)
#lat_b = (53.410, 53.415)



# Plot
plt.xlim(lon_b)
plt.ylim(lat_b)
print(nds.shape)
# plt.savefig("data/figures/dublin_south_clean.jpeg", dpi=300)
plt.show()


# In[1]


# con_graph = nx.read_gpickle("data/subgraphs/city_centre_clean.gpickle")

while nx.number_connected_components(con_graph) > 1:
    for component in list(nx.connected_components(con_graph)):
        if len(component) <= min([len(l) for l in list(nx.connected_components(con_graph))]):
            for node in component:
                print("node")
                con_graph.remove_node(node)

# Another graph/gdf reset
con_graph = momepy.gdf_to_nx(eds)
nds, eds = momepy.nx_to_gdf(con_graph)

# Add coordinates to eds
eds['start_coord'] = np.array(con_graph.edges)[:, 0]
eds['end_coord'] = np.array(con_graph.edges)[:, 1]

# Finally, calculate the distance matrix
indices = np.arange(np.array(list(con_graph.nodes)).shape[0])

# Pick random IDs and get their coordinates (for networkx)
sample_ids = np.random.choice(indices, 500, replace=False)
sample_coords = np.array(list(con_graph.nodes))[sample_ids]


# Formatting cordinates for dataframe columns and indexes later
coord_list = sample_coords.tolist()

for i in range(len(coord_list)):
    coord_list[i] = tuple(coord_list[i])

# Create empty distance_matrix and speed_matrix
N = len(sample_coords)
dists = np.zeros((N,N))
distance_matrix = pd.DataFrame(data=dists, index=sample_ids, columns=sample_ids)
speed_matrix, geom_matrix, highway_matrix = distance_matrix.copy(), distance_matrix.copy(), distance_matrix.copy()

# count for tracking
count = 0

# Get depot_coord and id
from scipy.spatial import distance

# Function to find closest point to another point from list of points
def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]

# Set a corner for depot (for example bottom left, top right)
corner = (np.max(sample_coords[:, 0]), np.max(sample_coords[:, 1]))

# Get depot coordinates and position in current points
depot = closest_node(corner, sample_coords)
d_idx = np.where(sample_coords == depot)[0][0]

# Set depot to beginning of sample points
sample_coords[[0, d_idx]] = sample_coords[[d_idx, 0]]
depot_coord, depot_point = sample_coords[0], sample_ids[0]

# every node pair combination
pairs = np.array(list(itertools.combinations(sample_coords, 2)))
id_pairs = np.array(list(itertools.combinations(sample_ids, 2)))


# loop through node pairs, find path and path_length for each
for i in range(len(pairs)):

    # get pair and IDs
    pair = pairs[i]
    id_pair = id_pairs[i]
    nodes = np.array(list(con_graph.nodes))[id_pair]

    # networkx shortest path
    path = np.array(nx.shortest_path(con_graph, tuple(nodes[0]), tuple(nodes[1]), weight='length'))

    # Only include path if none of the other selected nodes lie along it
    if (depot_coord in pair) or (len([i for i in path if i in sample_coords]) == 2):

        # read speeds from each edge along path
        lengths, speeds, geoms, highways = [], [], [], []

        for j in range(len(path) - 1):

            row = eds[((eds['start_coord']==tuple(path[j])) | (eds['end_coord']==tuple(path[j]))) &
                         ((eds['start_coord']==tuple(path[j + 1])) | (eds['end_coord']==tuple(path[j + 1])))]

            lengths.append(row['length'].values[0])
            speeds.append(row['maxspeed'].values[0])
            geoms.append(row['geometry'].values[0])
            highways.append(row['highway'].values[0])

        # get path length, add to matrix
        distance_matrix.loc[id_pair[0], id_pair[1]] = np.sum(np.array(lengths))

        # record average speed for path
        avg_speed = np.sum((np.array(speeds) * np.array(lengths)) / np.sum(np.array(lengths)))
        speed_matrix.loc[id_pair[0], id_pair[1]] = avg_speed

        # get total path geometry
        newline = geometry.MultiLineString(geoms)
        newline = ops.linemerge(newline)
        geom_matrix.loc[id_pair[0], id_pair[1]] = newline

        # find most common road type along path
        highway_matrix.loc[id_pair[0], id_pair[1]] = most_frequent(highways)

    # impromptu progress bar
    count += 1
    if count % 1000 == 0:
        print(count // 1000)


# Diagnostic check for sparsity
x = distance_matrix[distance_matrix != 0].values.flatten()
len(x[~np.isnan(x)]) / len(pairs)


# Need directionality in the matrices, just add tranposes to bottom triangle
distance_matrix = distance_matrix + distance_matrix.T
speed_matrix = speed_matrix + speed_matrix.T

for i in range(geom_matrix.shape[0]):
    for j in range(i, geom_matrix.shape[1]):
        geom_matrix.iloc[j, i] = geom_matrix.iloc[i, j]
        highway_matrix.iloc[j, i] = highway_matrix.iloc[i, j]

# Add cordinates as columns and indices
distance_matrix.index, distance_matrix.columns = coord_list, coord_list
speed_matrix.index, speed_matrix.columns = coord_list, coord_list
geom_matrix.index, geom_matrix.columns = coord_list, coord_list
highway_matrix.index, highway_matrix.columns = coord_list, coord_list

# Look at how sparse it is
distance_matrix.to_pickle("data/distance_matrices/dublin_south/corner_n500.pkl")
speed_matrix.to_pickle("data/speed_matrices/dublin_south/corner_n500.pkl")
geom_matrix.to_pickle("data/geom_matrices/dublin_south/corner_n500.pkl")
highway_matrix.to_pickle("data/highway_matrices/dublin_south/corner_n500.pkl")
