def create_instance(domain, n, depot, traffic, rain):

    import pandas as pd
    import numpy as np
    import src.generate_grid.grid as grid

    # Scat site testing data
    ds = pd.read_pickle("data/scats_sites_with_elev.pkl")
    ds = ds.loc[:, "Lat":"Elev"]
    #ds = ds.append(ds1)

    # Clean data up a little
    ds = ds.drop(ds[ds['Long']==0].index)
    ds = ds.drop(ds[ds['Lat']==0].index)
    ds = ds.drop(ds[ds['Long']>-6].index)
    ds = ds.drop(ds[ds['Long']<-6.5].index)
    ds = ds.drop(ds[ds['Lat']>53.5].index)
    ds = ds.drop(ds[ds['Lat']<53.1].index)
    ds = ds[["Long", "Lat", "Elev"]]

    # round values to grid values
    epoints = np.round(ds.to_numpy(), 4)

    # Bounding box for Dublin
    # Round to 4 decimals points otherwise not enough memory for grid
    lon_b = (-6.5, -6)
    lat_b = (53.1, 53.5)

    # Step size
    h = 0.0001


    # Make the Grid
    dublin = grid.Grid(lon_b=lon_b, lat_b=lat_b, h=h)

    v_file = "{}/{}_n{}.pkl".format(domain, depot, n)

    # add points to grid
    dublin.add_elevation_points(epoints, filename="data/elevation_matrices/{}".format(v_file))
    dublin.create_interpolation(k=6, p=2)

    # Vertices
    vdf = pd.read_pickle("data/distance_matrices/{}".format(v_file))
    vpoints = list(vdf.columns)
    vpoints = np.round(np.array(vpoints), 4)

    # We can get the elevations directly from open elevation instead of interpolating (TODO: ask others how to do this)
    vdf = pd.DataFrame(vpoints, columns=['longitude', 'latitude'])
    #vdf.to_pickle("data/instance_elevs/n20/n20_lat_long.pkl")
    #create_elev_query_file("data/instance_elevs/n20/n20_lat_long.pkl", "data/instance_elevs/n20/n20_to_query.json")
    dublin.add_vertices(vpoints)

    # create df for grid
    dublin.create_df()

    # compute matrices for various edges
    dublin.compute_distance(mode="OSM", filename="data/distance_matrices/{}".format(v_file))
    dublin.compute_gradient()
    dublin.read_driving_cycle("data/WLTP.csv", h=4, hbefa_filename="data/HBEFA_Driving_Cycles.pkl")
    dublin.compute_speed_profile(filename="data/speed_matrices/{}".format(v_file))
    dublin.create_geometries("data/geom_matrices/{}".format(v_file))

    dublin.compute_traffic(filename="data/traffic_matrices/{}.pkl".format(traffic))
    dublin.read_highways(filename="data/highway_matrices/{}".format(v_file))
    dublin.compute_level_of_service()

    dublin.read_weather(filename="data/weather_matrices/{}.pkl".format(rain))
    dublin.compute_weather_correction()
    dublin.read_skin_temp(filename="data/weather_matrices/Skin_temperature_averages.pkl")

    dublin.compute_cost(method="copert with meet")
    np.set_printoptions(suppress=True)

    return dublin
