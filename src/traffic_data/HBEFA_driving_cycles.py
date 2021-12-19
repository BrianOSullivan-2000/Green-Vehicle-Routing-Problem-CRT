
# Save all the HBEFA Driving Cycles using Multi Indexing

import pandas as pd
import numpy as np

# The indices are road class, speed limit, and Level of Service (LoS)
arrays = [["motorway", "trunk", "primary", "secondary", "tertiary", "residential"],
            [130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30], [1, 2, 3]]

# Create Multi Index
index = pd.MultiIndex.from_product(arrays, names=["Highway", "Speed Limit", "LoS"])

# Read in Average Velocities and Stop Percentages and clean them up
df = pd.read_csv("data/HBEFA_Driving_Cycles.csv")
avg_velocities = df.iloc[:, 3::5].values.T.flatten()
stop_percentages = df.iloc[:, 4::5].values.T.flatten()

# Create multi index dataframe and save for later
HBEFA_Driving_Cycles = pd.DataFrame(data = {"Velocity":avg_velocities, "Stop %":stop_percentages}, index=index)
HBEFA_Driving_Cycles.to_pickle("data/HBEFA_Driving_Cycles.pkl")
pd.read_pickle("data/HBEFA_Driving_Cycles.pkl")
