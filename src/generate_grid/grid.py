import random
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, distance_matrix
import math
import utm


class Grid():
    """ The grid class is a framework for combining geospatial data from
    different datasets (elevations, routing, traffic, etc.). When initialised,
    an NxN grid with step size h is created with cartesian coordinates.
    Vertices are placed on to the grid using an array of (x,y) points.
    Edges are defined using pandas dataframes, i.e. a distance, gradient,
    speed and cost matrix between all vertices. Costs are CO2 emissions
    along each path and can be computed using either MEET or COPERT methodologies.
    """


    def __init__(self, lon_b, lat_b, h=1):
        """ Arguments
            lon_b, lat_b - bounding box for latitude and longitude, given as tuples
            h - Step size between points of grid
            load - Boolean, determine if MEET load factor is included or not
        """

        # Base grid defined as self.xx and self.yy
        self.x, self.y = np.arange(lon_b[0], lon_b[1], h), np.arange(lat_b[0], lat_b[1], h)
        self.x, self.y = np.round(self.x, 4), np.round(self.y, 4)
        self.yy, self.xx = np.meshgrid(self.y, self.x)

        # Placeholders made for elevation and vertex values
        # self.elevation and self.vertice are NxN matrices
        # following the shape of the grid
        self.elevation = np.zeros((len(self.x), len(self.y)))
        self.vertice = np.zeros((len(self.x), len(self.y)), dtype=int)

        # Call in coefficients for both methodologies
        self.MEETdf = pd.read_csv("data\MEET_Slope_Correction_Coefficients_Light_Diesel_CO2.csv")
        self.COPERTdf = pd.read_csv("data\Copert_GVRP_Coefficients.csv")

        # Convert grid coordinates to array of (x, y) points
        self.points = np.append(self.xx.reshape(-1,1), self.yy.reshape(-1,1), axis=1)




    def add_elevation_points(self, data):
        """ Add elevation values to points as described by input data

        data - numpy array of points, points have shape (x, y, elevation)
        """

        # Loop through each input datapoint and add it's
        # elevation to the corresponding gridpoint
        for i in data:

            # Find corresponding index for lon, lat
            index = (np.where(self.x == i[0])[0], np.where(self.y == i[1])[0])

            if np.all(index):
                self.elevation[index[0][0], index[1][0]] = i[2]




    def add_vertices(self, data):
        """ Define specific points as vertices as described by input data

        data - numpy array of points, points have shape (x, y)
        """

        # Loop through each input datapoint and add it
        # as a vertex to grid with a unique index
        self.vertice_count = 1
        for i in data:

            # Find corresponding index for lon, lat
            index = (np.where(self.x == i[0])[0], np.where(self.y == i[1])[0])

            # condition if index isn't found in grid
            if ((len(index[0]) == 1) and (len(index[1]) == 1)):

                self.vertice[index[0][0], index[1][0]] = self.vertice_count
                self.vertice_count += 1




    def create_interpolation(self, data, k=10, p=2):
        """ Take in input data and interpolate elevations over entire grid.
            Interpolation carried out with IDW2 method.

            data - numpy array of points, points have shape (x, y, elevation)
        """
        import scipy.interpolate as interp

        # Get lats and lons and convert to grid with utm projection
        lons, lats = data[:, 0], data[:, 1]

        # Convert both input points and gridpoints to correct grid using UTM projection
        obs_raw = np.asarray(utm.from_latlon(np.asarray(lats), np.asarray(lons))[0:2])
        obs = np.stack((obs_raw[0], obs_raw[1]), axis=1)
        grid_obs_raw = np.asarray(utm.from_latlon(self.yy.ravel(), self.xx.ravel())[0:2])
        grid_obs = np.stack((grid_obs_raw[0], grid_obs_raw[1]), axis=1)

        # Object for querying nearest points
        tree = cKDTree(np.array(obs))

        # Each point in grid determines k=10 nearest points from input data
        # d is separation distances and inds are indexes of corresponding points
        d, inds = tree.query(np.array(grid_obs), k=k)

        # Calculate IDW2 weights
        w = 1.0 / d**p

        # Compute weighted averages for each gridpoint
        weighted_averages = np.sum(w * data[:, 2][inds], axis=1) / np.sum(w, axis=1)

        # Set elevation to each gridpoint
        self.elevation = np.reshape(weighted_averages, (len(self.x), len(self.y)))

        # Input datapoints are not included in interpolation
        # input them manually
        self.add_elevation_points(data)




    def create_df(self):
        """ Create pandas DataFrame with positions, gradient and vertex status
            Needed for later geopandas manipulation, and to determine various
            edge matrices (distance, gradient, cost, etc.)
        """

        # Generate input data
        data = {'x':self.points[:,0], 'y':self.points[:,1],
                'elevation':self.elevation.flatten(), 'is_vertice':self.vertice.flatten()}

        # Create DataFrame
        self.df = pd.DataFrame(data=data)

        # Order vertices
        self.df.loc[self.df['is_vertice'] != 0, 'is_vertice'] = np.arange(1,self.df.loc[self.df['is_vertice'] != 0, 'is_vertice'].shape[0]+1)
        self.depot = self.df.loc[self.df['is_vertice'] == 1]




    def create_geometries(self, filename):
        """ Read previously prepared geometries of edges from pickle file

            filename - path to file
        """

        self.geom_matrix = pd.read_pickle(filename)




    def compute_distance(self, mode="Euclidean", filename=None):
        """ Create distance matrix between all vertices.
            Represented as pandas DataFrame

            mode - distance metric used, currently only Euclidean available
        """

        # Identify all vertices in grid
        data = self.df[self.df['is_vertice'] != 0].values

        if mode == "Euclidean":

            # Find great circle distances from lon/lat values
            lons = data[:, 0]
            lats = data[:, 1]

            # Syntax to make relationship matrix for all point pairs
            dx_sub = lons[..., np.newaxis] - lons[np.newaxis, ...]
            dy_sub = lats[..., np.newaxis] - lats[np.newaxis, ...]
            dy_add = lats[..., np.newaxis] + lats[np.newaxis, ...]

            # Point pair separations in x and y
            dx = (dx_sub * 40075 * np.cos((dy_add) * math.pi / 360) / 360) * 1000
            dy = (dy_sub * 40075 / 360) * 1000

            # Pythagoras to make final net separations
            d = np.array([dx,dy])
            self.distance_matrix = pd.DataFrame((d**2).sum(axis=0)**0.5,
                                                index=data[:, 3].astype(int),
                                                columns=data[:, 3].astype(int))

        elif mode == "Manhattan":

            # Find great circle distances from lon/lat values
            lons = data[:, 0]
            lats = data[:, 1]

            # Syntax to make relationship matrix for all point pairs
            dx_sub = lons[..., np.newaxis] - lons[np.newaxis, ...]
            dy_sub = lats[..., np.newaxis] - lats[np.newaxis, ...]
            dy_add = lats[..., np.newaxis] + lats[np.newaxis, ...]

            # Point pair separations in x and y
            dx = (dx_sub * 40075 * np.cos((dy_add) * math.pi / 360) / 360) * 1000
            dy = (dy_sub * 40075 / 360) * 1000

            # Just sum dx and dy for Manhattan distance
            d = np.array([dx,dy])
            self.distance_matrix = pd.DataFrame((d).sum(axis=0),
                                                index=data[:, 3].astype(int),
                                                columns=data[:, 3].astype(int))


        elif mode == "OSM":

            # Read distance matrix prepper from process_osm.py
            self.distance_matrix = pd.read_pickle(filename)




    def compute_gradient(self):
        """ Compute gradients between all vertices. Represented as pandas DataFrame
            Gradient defined as Rise/Run in %
        """

        # Identify all vertices in grid
        data = self.df[self.df['is_vertice'] != 0].values

        # Vertex elevations
        elevs = data[:, 2]

        # Compute rise and run for all point pairs
        # if distance matrix is non-euclidean, use euclidean values for run
        rise = elevs[..., np.newaxis] - elevs[np.newaxis, ...]
        run = self.distance_matrix.to_numpy()

        gf = rise/run
        gf[gf == (np.inf)] = 0
        gf[gf == (-1 * (np.inf))] = 0

        # Create dataframe of rise/run
        self.gradient_matrix = pd.DataFrame(gf,
                                            index=data[:, 3].astype(int),
                                            columns=data[:, 3].astype(int))

        # NA values along diagonal from TrueDivide error
        self.gradient_matrix = self.gradient_matrix.fillna(0)




    def read_driving_cycle(self, filename, h, hbefa_filename=None):
        """ Read in driving cycle from .csv file. Used to read in WLTP,
            but header is adjustable for potential use of other driving cycles.
            Driving cycles saved as pandas DataFrames.

            filename - path to driving cycle .csv file
            h - line in .csv file to be used as header
        """

        # Read in driving cycle as net data and second-by-second data
        self.dc_net = pd.read_csv(filename, header=h).iloc[:, 6:16].iloc[0:4]
        self.dc_raw = pd.read_csv(filename, header=h).iloc[1:, 0:6]

        # Divide driving cycle into different modes according to phase
        self.dc_urban = self.dc_raw[self.dc_raw['Phase']=='Low']
        self.dc_suburban = self.dc_raw[self.dc_raw['Phase']=='Middle']
        self.dc_rural = self.dc_raw[self.dc_raw['Phase']=='High']
        self.dc_highway = self.dc_raw[self.dc_raw['Phase']=='Extra-high']


        # Read in HBEFA driving cycles (These are the ones we'll actually use)
        if hbefa_filename:
            self.dc_HBEFA = pd.read_pickle(hbefa_filename)




    def read_weather(self, filename, h=0.005):
        """ Read in weather data as GeoDataFrame of points. A low-resolution grid
            is interpolated over the region of interest for the weather data.
            Evenly spaced rectangles are computed about each point with the
            weather variables assigned. (Currently only total precipitation)

            filename - path to weather data, rainfall units assumed as m/hour
                       with label 'tp' (Total Precipitation),
                       coordinate columns assumed "longitude" and "latitude",
                       file type is pickle (.pkl)

            h - step size of grid to interpolate
        """

        # Import Polygon function for rectangles
        from shapely.geometry import Polygon

        # Read in file and convert to mm
        self.weather = pd.read_pickle(filename)
        self.weather.loc[:, 'tp'] = self.weather.loc[:, 'tp'] * 1000

        lons, lats, prec = self.weather['longitude'], self.weather['latitude'], self.weather['tp']

        # Create gridpoints for interpolation array
        x, y = np.arange(self.x[0], self.x[-1], h), np.arange(self.y[0], self.y[-1], h)
        yy, xx = np.meshgrid(y, x)

        # Convert both input points and gridpoints to grid using UTM projection
        obs_raw = np.asarray(utm.from_latlon(np.asarray(lons), np.asarray(lats))[0:2])
        obs = np.stack((obs_raw[0], obs_raw[1]), axis=1)
        grid_obs_raw = np.asarray(utm.from_latlon(xx.ravel(), yy.ravel())[0:2])
        grid_obs = np.stack((grid_obs_raw[0], grid_obs_raw[1]), axis=1)

        # IDW2 interpolation
        tree = cKDTree(np.array(obs))

        # Get distances, indices, compute weights, find final weighted averages
        d, inds = tree.query(np.array(grid_obs), k=7)
        w = 1.0 / d**2
        weighted_averages = np.sum(w * prec.values[inds], axis=1) / np.sum(w, axis=1)

        # Total precipitation array
        tp = np.reshape(weighted_averages, (len(x), len(y)))

        # Geometry of interpolated array as points, then gather data
        geometry = gpd.points_from_xy(xx.flatten(), yy.flatten())
        names = {'Precipitation':tp.flatten(), 'longitude':xx.flatten(), 'latitude':yy.flatten()}

        # Finally, interpolated array is converted to GeoDataFrame
        gdf = gpd.GeoDataFrame(pd.DataFrame(data=names), columns=['Precipitation'],
                                 geometry=geometry, crs={'init' : 'epsg:4326'})


        # Find midpoints between all gridpoints of GeoDataFrame (These are the rectangle corners)
        x_mid, y_mid = np.arange(x[0] - (h/2), x[-1] + 2*(h/2), h), np.arange(y[0] - (h/2), y[-1] + 2*(h/2), h)

        # Starting points and trackers for loops
        x0, y0 = x_mid[0], y_mid[0]
        x_step, y_step = x0, y0

        # List of rectangles
        recs = []

        # Loop over longitude
        for i in range(len(x)):
            # Reset latitude for each new longitude
            y_step = y0

            # Loop over latitude
            for j in range(len(y)):
                # Get all four corners of rectangle surrounding point
                recs.append([(x_step, y_step), (x_step+h, y_step), (x_step+h, y_step+h), (x_step, y_step+h)])

                # Move on to next rectangle by step size
                y_step += h
            x_step += h


        # Set rectangles as new geometry for GeoDataFrame
        grid_geom = pd.Series(recs).apply(lambda x: Polygon(x))
        gdf['geometry'] = grid_geom


        # Bin values according to rainfall ranges
        # Two different options, m50 study by De Courcy et al
        # or London study by Tsapakis et al
        #m50_bins = np.array((0, 0.0005, 0.5, 4, 50))
        london_bins = np.array((0, 0.0005, 0.2, 6, 50))

        #m50_vals = np.array((1, 1-0.025, 1-0.053, 1-0.155))
        london_vals = np.array((1, 1-0.021, 1-0.038, 1-0.06))

        # Going with london metrics for now
        gdf['Rain_Correction'] = np.digitize(gdf['Precipitation'], london_bins)
        gdf['Rain_Correction'] = np.array([london_vals[idx] for idx in gdf['Rain_Correction']])

        # Final output of GeoDataFrame
        self.weather = gdf




    def read_skin_temp(self, filename):
        """ Interpolate skin temperature for grid. Temperature is
            only interpolated at depot location for cold starts.

            filename - path to average hourly temperature data in degrees
                       Celsius with label 'skt' (Skin Temperature),
                       coordinate columns assumed "longitude" and "latitude",
                       file type is pickle (.pkl)
        """

        # Read in file
        self.temp = pd.read_pickle(filename)
        self.temp.loc[:, 'skt'] = self.temp.loc[:, 'skt'] - 273.15

        # Get depot location and temperature points
        depot = self.depot.iloc[:, 0:2].values[0]
        lons, lats, skt = self.temp['longitude'], self.temp['latitude'], self.temp['skt']

        # Convert both input points and gridpoints to grid using UTM projection
        obs_raw = np.asarray(utm.from_latlon(np.asarray(lons), np.asarray(lats))[0:2])
        obs = np.stack((obs_raw[0], obs_raw[1]), axis=1)
        grid_obs_raw = np.asarray(utm.from_latlon(depot[0], depot[1]))
        grid_obs = np.stack((grid_obs_raw[0], grid_obs_raw[1]))

        # IDW2 interpolation
        tree = cKDTree(np.array(obs))

        # Get distances, indices, compute weights, find final weighted averages
        d, inds = tree.query(np.array(grid_obs), k=len(skt))
        w = 1.0 / d**2
        self.depot_temp = np.sum(w * skt.values[inds]) / np.sum(w)




    def compute_weather_correction(self):
        """ This function gets the overall rainfall levels over the
            line geometry of each edge using the self.weather grid.

            Output is saved as velocity correction factor along
            each edge caused by rainfall.
        """

        # Read in shape of output matrix
        self.weather_correction_matrix = self.geom_matrix.copy()

        # Iterate over each row of matrix
        for index, row in self.geom_matrix.iterrows():

            # list of weights for each row
            net_weights = []

            # For each line in row
            for line in row:
                # Only compute if line exists
                if line != 0:

                    # Find all intersecting rectanges in weather grid
                    inter_recs = self.weather[self.weather.intersects(line)]

                    if inter_recs.shape[0] > 0:
                        # Only bother computing if there is variance along line
                        if not (inter_recs['Rain_Correction']==inter_recs['Rain_Correction'].iloc[0]).all():

                            # Get total length and length weights
                            total_len, len_weights = line.length, []

                            # find proportion of line within each intersecting rectangle
                            for rec in inter_recs['geometry']:
                                len_weights.append(rec.intersection(line).length / total_len)

                            # Sum up contributions from each rectangle for final correction factor
                            net_weights.append(np.sum(np.array(len_weights) * inter_recs['Rain_Correction']))

                        # Correction factor is constant, no need for computation
                        else:
                            net_weights.append(inter_recs['Rain_Correction'].iloc[0])

                    else:
                        print("Rectangle Error")
                        self.line = line_gdf
                        net_weights.append(0)

                # Include zeros to maintain shape of matrix
                else:
                    net_weights.append(0)

            # Adjust row with rainfall correction factors
            self.weather_correction_matrix[index] = net_weights




    def compute_speed_profile(self, filename=None):
        """ Compute average velocity along paths between all vertices.
            Represented as pandas DataFrame. Average velocity currently accessed
            from driving cycle according to distance along path.
        """

        # Average velocities from driving cycle
        velocity = self.dc_net['v_ave without stops (km/h)']

        if filename==None:

            # Separations and speed_matrix
            dists = self.distance_matrix.to_numpy()
            speeds = np.empty(dists.shape)

            # Assuming average speed by separation of vertices
            # Final values from WLTP Driving Cycle
            speeds[(dists < 10000) & (dists > 1)] = velocity[0]
            speeds[(dists < 20000) & (dists > 10000)] = velocity[1]
            speeds[(dists < 50000) & (dists > 20000)] = velocity[2]
            speeds[(dists > 50000)] = velocity[3]

            # Need indices of vertices for labelling
            data = self.df[self.df['is_vertice'] != 0].values[:, 3].astype(int)

            # Create dataframe
            self.velocity_matrix = pd.DataFrame(speeds, index=data, columns=data)

        # Read previously prepared speed_matrix
        else:

            # Add zero to beginning of velocity options to correctly create bins
            velocity = list(velocity)
            velocity.insert(0, 0)

            # Create bins using midpoint between average velocities in driving cycles
            bins = [(velocity[i] + velocity[i+1])/2 for i in range(len(velocity)-1)]
            bins[0] = 0
            bins.append(200)

            # Read speed matrix and save dataframe shape, indexes and columns
            self.velocity_matrix, self.speed_limit_matrix = pd.read_pickle(filename), pd.read_pickle(filename)
            indices, columns = self.velocity_matrix.index, self.velocity_matrix.columns
            v_shape = self.velocity_matrix.shape

            # Bin the velocity matrix according to driving cycle speeds
            v_binned = np.array([velocity[idx] for idx in np.digitize(self.velocity_matrix.values.flatten(), bins)])

            # Convert back to original shape and save
            v_binned = v_binned.reshape(v_shape)
            self.velocity_matrix = pd.DataFrame(v_binned, index=indices, columns=columns)




    def compute_traffic(self, filename):
            """ Traffic situations affect both average speed and stop time percentage.
                Traffic levels are read from file and converted to Level of Service (LoS)
                based on velocity along edge and vehicle count/hr.

                filename - interpolated file (created in plotting_traffic_cover.py)
            """

            # Import Polygon function for rectangles
            from shapely.geometry import Polygon

            # Convert data to GeoDataFrame
            self.traffic_levels = pd.read_pickle(filename)
            geometry = gpd.points_from_xy(self.traffic_levels["longitude"].values,
                                          self.traffic_levels["latitude"].values)
            names = {'Traffic':self.traffic_levels["traffic"].values,
                    'longitude':self.traffic_levels["longitude"].values,
                    'latitude':self.traffic_levels["latitude"].values}

            gdf = gpd.GeoDataFrame(pd.DataFrame(data=names), columns=['Traffic'],
                                     geometry=geometry, crs={'init' : 'epsg:4326'})

            # Grid step size and points, note step size is assumed equal in x and y
            h = abs(np.unique(gdf['geometry'].x)[0] - np.unique(gdf['geometry'].x)[1])
            x = np.arange(gdf['geometry'].x.values[0], gdf['geometry'].x.values[-1], h)
            y = np.arange(gdf['geometry'].y.values[0], gdf['geometry'].y.values[-1], h)

            # Find midpoints between all gridpoints of GeoDataFrame (These are the rectangle corners)
            x_mid = np.arange(x[0] - (h/2), x[-1] + 2*(h/2), h)
            y_mid = np.arange(y[0] - (h/2), y[-1] + 2*(h/2), h)

            # Starting points and trackers for loops
            x0, y0 = x_mid[0], y_mid[0]
            x_step, y_step = x0, y0

            # List of rectangles
            recs = []

            # Loop over longitude
            for i in range(len(x) + 1):
                # Reset latitude for each new longitude
                y_step = y0

                # Loop over latitude
                for j in range(len(y) + 1):
                    # Get all four corners of rectangle surrounding point
                    recs.append([(x_step, y_step), (x_step+h, y_step), (x_step+h, y_step+h), (x_step, y_step+h)])

                    # Move on to next rectangle by step size
                    y_step += h
                x_step += h


            # Set rectangles as new geometry for GeoDataFrame
            grid_geom = pd.Series(recs).apply(lambda x: Polygon(x))
            gdf['geometry'] = grid_geom
            self.traffic = gdf




    def read_highways(self, filename):
        """ Read in highway matrix of road edges

            filename - path to highway matrix file
        """

        # Read in matrix
        self.highway_matrix = pd.read_pickle(filename)
        arr = self.highway_matrix

        # Convert all link roads to standard roads
        arr[arr == 'motorway_link'] = 'motorway'
        arr[arr == 'trunk_link'] = 'trunk'
        arr[arr == 'primary_link'] = 'primary'
        arr[arr == 'secondary_link'] = 'secondary'
        arr[arr == 'tertiary_link'] = 'tertiary'

        self.highway_matrix.loc[:, :] = arr




    def compute_level_of_service(self):
        """ Level of service (LoS) determined along each edge according to speed
            along edge and vehicle count. Proportion of each LoS is found along
            each edge, which is then used to calculate average velocity and
            stop time percentage as weighted averages.
        """

        # The velocities used to calculate LoS are the initial simplified
        # average velocities taken from the WLTP driving cycle

        # Create avg velocity and stop percentages matrix
        self.hbefa_velocity_matrix = self.velocity_matrix.copy()
        self.stop_percentages_matrix = self.velocity_matrix.copy()

        # Read in shape of output matrix
        self.level_of_service = self.geom_matrix.copy()

        # Read in highway/speed limit classification
        self.lund_vasteras_ranges = pd.read_csv("data/lund_vasteras_ranges.csv")

        # Highway options are made lowercase
        highway_types = np.char.lower(self.lund_vasteras_ranges.iloc[1:, 0].values.astype(str))

        # LoS and Driving Cycle speed limit ranges vary
        los_spl = np.array((50, 90, 110))
        dc_spl = np.array((30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130))

        # Get midpoints to find closest available value
        los_spl_bins = (los_spl[1:] + los_spl[:-1]) / 2
        dc_spl_bins = (dc_spl[1:] + dc_spl[:-1]) / 2

        # Get just ranges
        self.lund_vasteras_ranges = self.lund_vasteras_ranges.iloc[:, 1:]

        for idx, row in self.lund_vasteras_ranges.iterrows():
            self.lund_vasteras_ranges.loc[idx] = row.str.split(", ")
            self.lund_vasteras_ranges.loc[idx] = np.array(self.lund_vasteras_ranges.loc[idx])

        # Iterate over each row of matrix
        for idx, row in self.geom_matrix.iterrows():

            net_velocity, net_stop = [], []

            # For each line in row
            for line in row:
                # Only compute if line exists
                if line != 0:

                    # Find all intersecting rectanges in weather grid
                    inter_recs = self.traffic[self.traffic.intersects(line)]

                    if inter_recs.shape[0] > 0:

                        # Get column index from geom_matrix
                        col_idx = row[row == line].index[0]

                        # Get highway and speed of line (row, column index for lund/vasteras ranges)
                        highway = self.highway_matrix.loc[[idx], [col_idx]].values[0, 0]
                        highway_idx = np.where(highway_types == highway)[0][0]
                        spl = self.speed_limit_matrix.loc[[idx], [col_idx]].values
                        los_spl_idx = np.digitize(spl, los_spl_bins)[0][0]

                        # Get corresponging los bins and bin traffic levels for each line segment
                        los_bins = np.array(self.lund_vasteras_ranges.iloc[highway_idx, los_spl_idx]).astype(float)
                        los_segments = np.digitize(inter_recs['Traffic'].values, los_bins) + 1

                        # Get average speed and stop percentage for each segment
                        spl = self.speed_limit_matrix.loc[[idx], [col_idx]].values
                        dc_spl_idx = np.digitize(spl, dc_spl_bins)[0][0]

                        # Empty arrays for average velocity and stop percentage for each segment
                        avg_velocities = np.empty(los_segments.shape)
                        stop_percentages =  np.empty(los_segments.shape)

                        # Get average velocity and stop percentage from HBEFA driving
                        # cycles according to road class, speed limit, and LoS
                        for j in range(len(los_segments)):

                            dc = self.dc_HBEFA.loc[(highway, dc_spl[dc_spl_idx], los_segments[j])]
                            avg_velocities[j], stop_percentages[j] = dc['Velocity'], dc['Stop %'] / 100

                        # Get total length and length weights
                        total_len, len_weights = line.length, []

                        # find proportion of line within each intersecting rectangle
                        for rec in inter_recs['geometry']:
                            len_weights.append(rec.intersection(line).length / total_len)
                        len_weights = np.array(len_weights)

                        # Weighted average (by length) of velocity and stop percentage along edge
                        net_velocity.append(np.sum(np.array(len_weights) * avg_velocities))
                        net_stop.append(np.sum(np.array(len_weights) * stop_percentages))

                    else:
                        net_velocity.append(0)
                        net_stop.append(0)
                else:
                    net_velocity.append(0)
                    net_stop.append(0)

            self.hbefa_velocity_matrix[idx] = net_velocity
            self.stop_percentages_matrix[idx] = net_stop





    def compute_cost(self, method="MEET", idling=True, hbefa=True, load=True, rain=True, cold=True):
        """ Compute CO2 emitted along paths between all vertices using either
            MEET or COPERT methodologies. Represented as pandas DataFrame.
            Model coefficients from MEET or COPERT files previously read.

            method - input method for emissions model
            idling - option to include FMEFCM idling model
            load - option to include MEET load correction factor
        """

        # Identify all vertices in grid
        data = self.df[self.df['is_vertice'] != 0].values

        # Need distance and velocity matrices
        d = self.distance_matrix.to_numpy() / 1000

        # Using traffic velocities
        if hbefa == True:
            self.velocity_matrix = self.hbefa_velocity_matrix.copy()

        velocities = self.velocity_matrix.to_numpy()

        # Adjust velocities by rainfall correction factors
        if rain == True:
            velocities = velocities * self.weather_correction_matrix.to_numpy()

        # Round gradient matrix to values in [-6, -4, -2, 0, 2, 4, 6]
        grad = self.gradient_matrix.to_numpy() * 100
        grad[grad>6] = 6
        grad[grad<-6] = -6
        grad = (np.round(grad/2) * 2).astype(int)

        # MEET methodology
        if method.upper() == "MEET":

            if cold == True:

                temp_cf = (-0.0458 * self.depot_temp) + 1.9163
                cold_distance = (0.29 * velocities[0]) - 0.05
                beta = self.distance_matrix.iloc[0] / cold_distance
                speed_cf, a, omega = 1, 3.95, 153.36

                distance_cf = (1 - np.exp(-1*a*beta)) / (1 - np.exp(-1*a))

                EF_cold = omega * (speed_cf + temp_cf - 1) * distance_cf


            # Available gradients in model
            grads = [-6, -4, -2, 2, 4, 6]

            # Base model
            EF = (429.51 - 7.8227*velocities + 0.0617*(velocities**2))
            EF[0] = EF[0] + EF_cold

            # Slope correction factor coefficients
            cfs = self.MEETdf.loc[:, "A6":"Slope (%)"]

            # Loop through each grad, adjust EF if edge has specified grad
            for g in grads:

                # Find coefficients according to gradient
                cf = cfs.loc[cfs.index[cfs["Slope (%)"]==g][0]]

                # Velocities along edges with gradient g
                v = velocities[grad==g]

                # Adjust edges which correspond to gradient with coefficients
                EF[grad==g] = (cf[0]*v**6 + cf[1]*v**5 + cf[2]*v**4 +
                               cf[3]*v**3 + cf[4]*v**2 + cf[5]*v + cf[6]) * EF[grad==g]

                # Load correction factor
                if load:
                    EF[grad==g] = ((1.27) + (0.0614*g) + (-0.0011*g**2) +
                                   (-0.00235*v) + (-1.33/v)) * EF[grad==g]


        # Using COPERT model
        elif method.upper() == "COPERT":

            # LCVs do not have gradient or load correction options
            self.COPERTdf = self.COPERTdf.iloc[4:]

            # Grad = -6 returns negative EF
            grad[grad<-4] = -4
            grads = [-4, -2, 0, 2, 4, 6]
            EF = np.empty(d.shape)

            # Loop through each gradient
            for g in grads:

                # Velocities along edges with gradient g
                v = velocities[grad==g]

                # Read coefficients according to gradient (assuming 0% capacity)
                # Currently uses HGV
                cf = self.COPERTdf[(self.COPERTdf['Road.Slope'] == g/100) & (self.COPERTdf['Load'] == 0)].loc[:, "Alpha":"Hta"].values[0, :]

                # Calculate emissions factor (energy consumed in MJ/km)
                EF[grad==g] = (cf[0]*v**2 + cf[1]*v + cf[2] + cf[3]/v) / (cf[4]*v**2 + cf[5]*v + cf[6])

            # Cold start emissions
            if cold == True:
                # Cold quotient as described from COPERT report
                cold_quotient = 1.34 - (0.008 * self.depot_temp)

                # Average trip length from depot
                avg_len = np.mean(self.distance_matrix.iloc[1].values[1:]) / 1000

                # beta parameter (fraction of journey with cold starts)
                beta = 0.6474 - (0.02545*avg_len) - ((0.00974 - (0.000385*avg_len)) * self.depot_temp)

                # Cold start only included for edges leaving depot
                EF_cold = beta * EF[0] * (cold_quotient - 1)
                EF[0] = EF[0] + EF_cold

            # Convert to g/km from tables in Ntziachristos COPERT report
            EF = (EF/4.31) * 101 * 3.169


        elif method.upper() == "COPERT WITH MEET":

            # Get coefficients for LCV
            cf = self.COPERTdf.iloc[3]["Alpha":"Hta"]
            v = velocities

            # Calculate EF with no corrections
            EF = (cf[0]*v**2 + cf[1]*v + cf[2] + cf[3]/v) / (cf[4]*v**2 + cf[5]*v + cf[6])

            # Cold start emissions
            if cold == True:
                # Cold quotient as described from COPERT report
                cold_quotient = 1.34 - (0.008 * self.depot_temp)

                # Average trip length from depot
                avg_len = np.mean(self.distance_matrix.iloc[1].values[1:]) / 1000

                # beta parameter (fraction of journey with cold starts)
                beta = 0.6474 - (0.02545*avg_len) - ((0.00974 - (0.000385*avg_len)) * self.depot_temp)

                # Cold start only included for edges leaving depot
                EF_cold = beta * EF[0] * (cold_quotient - 1)
                EF[0] = EF[0] + EF_cold

            # Convert to g/km from tables in Ntziachristos COPERT report
            EF = (EF/4.31) * 101 * 3.169

            # Available gradients in model
            grads = [-6, -4, -2, 2, 4, 6]
            # Slope correction factor coefficients
            cfs = self.MEETdf.loc[:, "A6":"Slope (%)"]

            # Loop through each grad, adjust EF if edge has specified grad
            for g in grads:

                # Find coefficients according to gradient
                cf = cfs.loc[cfs.index[cfs["Slope (%)"]==g][0]]

                # Velocities along edges with gradient g
                v = velocities[grad==g]

                # Adjust edges which correspond to gradient with coefficients
                EF[grad==g] = (cf[0]*v**6 + cf[1]*v**5 + cf[2]*v**4 +
                               cf[3]*v**3 + cf[4]*v**2 + cf[5]*v + cf[6]) * EF[grad==g]

                # Load correction factor
                if load:
                    EF[grad==g] = ((1.27) + (0.0614*g) + (-0.0011*g**2) +
                                   (-0.00235*v) + (-1.33/v)) * EF[grad==g]


        else:
            print("Choose valid method")


        # Find total CO2 emitted over distance
        cost = EF * d
        self.EF = EF

        # Add idling costs
        if idling == True:

            # Find travel time along each edge without stops in seconds
            self.travel_times = (d / velocities) * 3600
            self.idling_times = self.travel_times.copy()

            # Using stop percentages from HBEFA traffic driving cycles
            if hbefa == True:
                self.idling_times = self.stop_percentages_matrix * self.travel_times
                self.idling_times = self.idling_times.to_numpy()

            else:
                # Average velocities from driving cycle, add lower bound for binning
                velocity = list(self.dc_net['v_ave without stops (km/h)'])

                # Create bins using midpoint between average velocities
                bins = [(velocity[i] + velocity[i+1])/2 for i in range(len(velocity)-1)]

                # Percentage of stoppage time according to speeds from driving cycles
                stop_percentages = [float(p[0:-1]) for p in self.dc_net['p_stop (%)']]

                v_binned = np.array([velocity[idx] for idx in np.digitize(velocities.flatten(), bins)])
                v_binned = np.reshape(v_binned, velocities.shape)
                v_binned[velocities==0] = 0

                # For each percentage
                for i in range(len(velocity)):

                    # Compute total time stopped as percentage of total time travelling
                    self.idling_times[v_binned==velocity[i]] = (stop_percentages[i] / 100) * self.travel_times[v_binned==velocity[i]]


            # Add costs from idling (0.4617g/s CO2 emitted while idling)
            cost = cost + (self.idling_times * 0.4617)


        # Round to nearest gram, make dataframe
        cost = np.round(cost)

        self.cost_matrix = pd.DataFrame(cost,
                                        index=data[:, 3].astype(int),
                                        columns=data[:, 3].astype(int))

        self.cost_matrix = self.cost_matrix.fillna(0)
