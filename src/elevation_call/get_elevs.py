# Script for converting each instance graph into appropriate
# format for open elevation api query

import json
import pandas as pd
from src.elevation_call.create_evel_query_file import create_elev_query_file
from src.elevation_call.read_elev_query_result import read_elev_res

domains = ["dublin_centre", "dublin_south", "m50"]
depots = ["centre", "corner"]
ns = ["20", "50", "100", "200", "500", "1000"]


for domain in domains:
    for depot in depots:
        for n in ns:

            if domain == "dublin_south" and n == "1000":
                print("Ignore this combo")
            else:
                coord_file = "data/distance_matrices/{}/{}_n{}.pkl".format(domain, depot, n)
                elev_file = "data/elevation_api_queries/{}/{}_n{}.json".format(domain, depot, n)
                coords = np.array(list(pd.read_pickle(coord_file).columns))[:, [1, 0]]

                df = pd.DataFrame(coords, columns=["latitude", "longitude"])

                df.to_pickle("data/elevation_api_queries/coords.pkl")

                create_elev_query_file("data/elevation_api_queries/coords.pkl", elev_file)

pd.read_json("data/elevation_api_queries/m50/corner_n50.json")


# In[1]


# Script for open elevations api results

import os

for domain in domains:
    for depot in depots:
        for n in ns:

            query = "data/elevation_api_queries/{}/{}_n{}.json".format(domain, depot, n)
            out = "data/elevation_api_results/{}/{}_n{}.json".format(domain, depot, n)

            if domain == "dublin_south" and n == "1000":
                print("Ignore this combo")
            else:
                os.system('curl -X POST https://api.open-elevation.com/api/v1/lookup -H\
                          "Content-Type:application/json" -d "@{}" --output {}'.format(query, out))


# In[1]


for domain in domains:
    for depot in depots:
        for n in ns:

            result = "data/elevation_api_results/{}/{}_n{}.json".format(domain, depot, n)
            final = "data/elevation_matrices/{}/{}_n{}.pkl".format(domain, depot, n)

            if domain == "dublin_south" and n == "1000":
                print("Ignore this combo")
            else:
                read_elev_res(result, final)


# In[1]


import numpy as np
# Making more elevation points
lon_b = (-6.421, -6.082)
lat_b = (53.151, 53.342)
h = 0.01

x, y = np.arange(lon_b[0], lon_b[1], h), np.arange(lat_b[0], lat_b[1], h)
x, y = np.round(x, 4), np.round(y, 4)
yy, xx = np.meshgrid(y, x)
coords = np.append(xx.reshape(-1,1), yy.reshape(-1,1), axis=1)
coords.shape

df = pd.DataFrame(coords, columns=["latitude", "longitude"])
df.to_pickle("data/elevation_api_queries/coords.pkl")

create_elev_query_file("data/elevation_api_queries/coords.pkl", "data/elevation_api_queries/coords_query.json")

query = "data/elevation_api_queries/coords_query.json"
out = "data/elevation_api_results/coords_results.json"

os.system('curl -X POST https://api.open-elevation.com/api/v1/lookup -H\
          "Content-Type:application/json" -d "@{}" --output {}'.format(query, out))

read_elev_res(out, "data/elevation_matrices/coords1.pkl")
