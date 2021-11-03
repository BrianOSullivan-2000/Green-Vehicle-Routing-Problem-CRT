import random
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
from scipy.spatial import cKDTree, distance_matrix
import math


class Grid():
    """ The grid class is a framework for combining geospatial data from
    different datasets (elevations, routing, traffic, etc.). When initialised,
    an NxN grid with step size h is created with cartesian coordinates.
    Vertices are placed on to the grid using an array of (x,y) points.
    Edges are defined using pandas dataframes, i.e. a distance, gradient,
    speed and cost matrix between all vertices. Costs are CO2 emissions
    along each path and can be computed using either MEET or COPERT methodologies.
    """

    def __init__(self, lon_b, lat_b, h=1, load=True):
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
        self.COPERTdf = pd.read_csv("data\Copert_GVRP_Coefficients.csv").iloc[4:]

        # Load correction factor for MEET model
        self.load = load

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
            self.vertice[index[0][0], index[1][0]] = self.vertice_count
            self.vertice_count += 1


    def create_interpolation(self, data):
        """ Take in input data and interpolate elevations over entire grid.
            Interpolation carried out with IDW2 method.

        data - numpy array of points, points have shape (x, y, elevation)
        """

        # TODO: Interpolation doesn't have correct lat/lon projection

        # Object for querying nearest points
        tree = sc.spatial.cKDTree(data[:, 0:2])

        # Each point in grid determines k=10 nearest points from input data
        # d is separation distances and inds are indexes of corresponding points
        d, inds = tree.query(np.c_[self.xx.ravel(), self.yy.ravel()], k=10)

        # Calculate IDW2 weights
        w = 1.0 / d**2

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


    def compute_distance(self, mode="Euclidean"):
        """ Create distance matrix between all vertices.
            Represented as pandas DataFrame

            mode - distance metric used, currently only Euclidean available
            TODO: add Manhattan and Haversine distance metrics
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

        if mode == "Manhattan":

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


    def compute_gradient(self):
        """ Compute gradients between all vertices. Represented as pandas DataFrame
            Gradient defined as Rise/Run in %
        """

        # Identify all vertices in grid
        data = self.df[self.df['is_vertice'] != 0].values

        # Vertex elevations
        elevs = data[:, 2]

        # Compute rise and run for all point pairs
        # TODO: (RUN IS ASSUMING EUCLIDEAN DISTANCE MATRIX)
        # if distance matrix is non-euclidean, use euclidean values for run
        rise = elevs[..., np.newaxis] - elevs[np.newaxis, ...]
        run = self.distance_matrix.to_numpy()

        # Create dataframe of rise/run
        self.gradient_matrix = pd.DataFrame(rise/run,
                                            index=data[:, 3].astype(int),
                                            columns=data[:, 3].astype(int))

        # NA values along diagonal from TrueDivide error
        self.gradient_matrix = self.gradient_matrix.fillna(0)


    def read_driving_cycle(self, filename, h):
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


    def compute_speed_profile(self):
        """ Compute average velocity along paths between all vertices.
            Represented as pandas DataFrame. Average velocity currently accessed
            from driving cycle according to distance along path.
        """

        # Average velocities from driving cycle
        velocity = self.dc_net['v_ave with stops (km/h)']

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


    def compute_cost(self, method="MEET"):
        """ Compute CO2 emitted along paths between all vertices using either
            MEET or COPERT methodologies. Represented as pandas DataFrame.
            Model coefficients from MEET or COPERT files previously read.

            method - input method for emissions model
        """

        # Identify all vertices in grid
        data = self.df[self.df['is_vertice'] != 0].values

        # Need distance and velocity matrices
        d = self.distance_matrix.to_numpy() / 1000
        velocities = self.velocity_matrix.to_numpy()

        # Round gradient matrix to values in [-6, -4, -2, 0, 2, 4, 6]
        grad = self.gradient_matrix.to_numpy() * 100
        grad[grad>6] = 6
        grad[grad<-6] = -6
        grad = (np.round(grad/2) * 2).astype(int)

        # MEET methodology
        if method.upper() == "MEET":

            # Available gradients in model
            grads = [-6, -4, -2, 2, 4, 6]

            # Base model
            EF = (429.51 - 7.8227*velocities + 0.0617*(velocities**2))

            # Slope correction factor coefficients
            cfs = self.MEETdf.loc[:, "A6":"Slope (%)"]

            # Loop through each grad, adjust EF is edge has specified grad
            for g in grads:

                # Find coefficients according to gradient
                cf = cfs.loc[cfs.index[cfs["Slope (%)"]==g][0]]

                # Velocities along edges with gradient g
                v = velocities[grad==g]

                # Adjust edges which correspond to gradient with coefficients
                EF[grad==g] = (cf[0]*v**6 + cf[1]*v**5 + cf[2]*v**4 +
                               cf[3]*v**3 + cf[4]*v**2 + cf[5]*v + cf[6]) * EF[grad==g]

                # Load correction factor
                if self.load:
                    EF[grad==g] = ((1.27) + (0.0614*g) + (-0.0011*g**2) +
                                   (-0.00235*v) + (-1.33/v)) * EF[grad==g]

            # Find total CO2 emitted rounded to nearest gram, make DataFrame
            cost = np.round(EF * d)
            self.cost_matrix = pd.DataFrame(cost,
                                            index=data[:, 3].astype(int),
                                            columns=data[:, 3].astype(int))

        # Using COPERT model
        elif method.upper() == "COPERT":

            # Grad = -6 returns negative EF (TODO: investigate this)
            grad[grad<-4] = -4
            grads = [-4, -2, 0, 2, 4, 6]
            EF = np.empty(d.shape)

            # Loop through each gradient
            for g in grads:

                # Velocities along edges with gradient g
                v = velocities[grad==g]

                # Read coefficients according to gradient (assuming 0% capacity)
                # Currently uses HGV
                # TODO: Investigate using LCV with MEET grad correction factor
                cf = self.COPERTdf[(self.COPERTdf['Road.Slope'] == g/100) & (self.COPERTdf['Load'] == 0)].loc[:, "Alpha":"Hta"].values[0, :]

                # Calculate emissions factor (energy consumed in MJ/km)
                EF[grad==g] = (cf[0]*v**2 + cf[1]*v + cf[2] + cf[3]/v) / (cf[4]*v**2 + cf[5]*v + cf[6])

            # Convert to g/km from tables in Ntziachristos COPERT report
            EF = (EF/4.31) * 101 * 3.169

            # Find total CO2 emitted rounded to nearest gram, make DataFrame
            cost = np.round(EF * d)

            self.cost_matrix = pd.DataFrame(cost,
                                            index=data[:, 3].astype(int),
                                            columns=data[:, 3].astype(int))

            self.cost_matrix = self.cost_matrix.fillna(0)

        # TODO: add FMEFCM idling rates to cost computation
        else:
            print("Choose valid method")
