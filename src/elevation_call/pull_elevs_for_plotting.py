# this script performs an elevation call
# for all gridpoints (0.001 lat long grid) within dublin bounds
# for plotting purposes

import pandas as pd
from src.elevation_call.create_evel_query_file import create_elev_query_file
from src.elevation_call.read_elev_query_result import read_elev_res
from src.generate_grid.grid import Grid
import math
import os

# dublin bounds
lon_b = (-6.5, -6)
lat_b = (53.1, 53.5)

# step size
h = 0.001

# create grid
dub_grid = Grid(lon_b=lon_b, lat_b=lat_b, h=h)

# number of batches for elev call
# as elev call can only handle 1500 at a time
batch_number = math.ceil(len(dub_grid.points)/1500)

# loop over batches
for i in range(batch_number):

    # get indices
    index_1 = 1500*i
    if i == batch_number - 1:
        index_2 = len(dub_grid.points) + 1
    else:
        index_2 = 1500*(i+1)

    # get current batch
    current = dub_grid.points[index_1:index_2]

    # make dataframe of points to query
    df = pd.DataFrame(current, columns=["longitude", "latitude"])
    # change order as required by api
    df = df[["latitude", "longitude"]]

    # set filepaths
    input = "data/elevation_call_for_plots/elev_dfs/elev_df_{}.pkl".format(i)
    query = "data/elevation_call_for_plots/elev_queries/elev_query_{}.json".format(i)
    out = "data/elevation_call_for_plots/elev_results/elev_result_{}.json".format(i)
    dfs_out = "data/elevation_call_for_plots/elev_dfs_out/elev_df_out_{}.pkl".format(i)

    # create input for creating query file
    df.to_pickle(input)

    # create query file
    create_elev_query_file(input, query)

    # perform api query
    os.system('curl -X POST https://api.open-elevation.com/api/v1/lookup -H\
              "Content-Type:application/json" -d "@{}" --output {}'.format(query, out))

    # process result to pickle
    read_elev_res(out, dfs_out)

# read in and stitch together all dfs created
elev_df = pd.DataFrame(columns=["latitude", "longitude", "elevation"])

for i in range(batch_number):
    filename = "data/elevation_call_for_plots/elev_dfs_out/elev_df_out_{}.pkl".format(i)
    df = pd.read_pickle(filename)
    elev_df = elev_df.append(df, ignore_index=True)

# save file
elev_df.to_pickle("data/elevation_call_for_plots/dub_grid_elev_001res.pkl")
