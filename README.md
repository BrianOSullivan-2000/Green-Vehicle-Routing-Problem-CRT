# Green-Vehicle-Routing-Problem-CRT

Group project for SFI CRT in Foundations of Data Science, Autumn 2021. 

We wish to create and characterise appropriate benchmark instances for solving the Green Vehicle Routing Problem, a problem which seeks to minimise carbon emissions in vehicle networks.
These instances will follow similar lines to traditional instances for CVRPs, but will instead cater towards an emissions model. Such instances will need to account for parameters such as gradient, vehicle load, traffic, and weather.

Supervised under Paula Carroll, Conor Sweeney and Mel Devine at University College Dublin.

________________________________________________
For navigating the repository:

- data: All datasets applied in the instance generation, including elevation datasets, traffic datasets, weather datasets, etc. Distance/speed limit/road class/geometry matrices for all benchmark instances generated are also included here.
- instances: The final benchmark instances. Instances range from 20-1000 nodes of different regions of Dublin's road network.
- src: All scripts used in data processing and instance generation. Each dataset is modelled in its respective folder. generate_grid/subgraph.py consists of filtering/cleaning of the road network. generate_grid/grid.py is where the final steps are taken for emissions calculations.
