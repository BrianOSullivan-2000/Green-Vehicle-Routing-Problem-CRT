import pandas as pd
import networkx as nx
import momepy
from src.elevation_call.create_evel_query_file import create_elev_query_file
from src.elevation_call.read_elev_query_result import read_elev_res

# load df
dublin_graph = nx.read_gpickle("data/dublin_graph.gpickle")

# nodes and info
nodes, edges, W = momepy.nx_to_gdf(dublin_graph, spatial_weights=True)

# read in n50 distance matrix
n50 = pd.read_csv("data/distance_matrices/n50.csv")

# isolate n50 node ids
n50_ids = pd.Series(n50.columns.values)
n50_ids = n50_ids.drop(0).reset_index(drop=True)
n50_ids = n50_ids.astype("int64")

# node lat and long
n50_geom = nodes["geometry"].loc[nodes.osmid.isin(n50_ids)]

n50_lat_long = pd.DataFrame({"latitude": n50_geom.y, "longitude": n50_geom.x})
n50_lat_long = n50_lat_long.reset_index(drop=True)

n50_lat_long.to_pickle("data/instance_elevs/n50/n50_lat_long.pkl")

create_elev_query_file("data/instance_elevs/n50/n50_lat_long.pkl", "data/instance_elevs/n50/n50_to_query.json")

read_elev_res("data/instance_elevs/n50/n50_query_results.json", "data/instance_elevs/n50/n50_lat_long_elev.pkl")

# TODO comment better
