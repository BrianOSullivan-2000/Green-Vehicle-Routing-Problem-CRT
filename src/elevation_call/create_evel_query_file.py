# script to create json file which will be used to look up elevations
# for multiple points through open-elevation's api
# working directory assumed project root

import json
import pandas as pd


def create_elev_query_file(input_filepath, output_filepath):
    # load position data
    # should be df with lat and long in cols
    post_data = pd.read_pickle(input_filepath)

    # create dict of data formatted as required for API call
    call_file_data = {}
    call_file_data["locations"] = post_data.to_dict("records")

    # write json file for use in query
    with open(output_filepath, 'w') as f:
        json.dump(call_file_data, f)
