# script to read in json file which contains elevations from open-api scrape
# and convert to a pandas dataframe and save
# working directory assumed project root

import json
from pandas import DataFrame

# open json file with API call results
# save to a dict
with open(".\\data\\elev_call_results.json", "r") as f:
    elev_call_data = json.load(f)

# convert dict to pandas dataframe
# with lat, long, elevation as cols
elevation_df = DataFrame.from_dict(elev_call_data["results"])

# save as pickle
elevation_df.to_pickle(".\\data\\post_and_elev.pkl")
