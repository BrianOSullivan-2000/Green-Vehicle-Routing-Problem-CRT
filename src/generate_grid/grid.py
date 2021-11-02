import random
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
from scipy.spatial import cKDTree, distance_matrix


class Grid():
    """ The grid class is a framework for combining geospatial data from
    different datasets (elevations, routing, traffic, etc.). When initialised,
    an NxN grid with step size h is created with cartesian coordinates.
    Vertices are placed on to the grid using an array of (x,y) points.
    Edges are defined using pandas dataframes, i.e. a distance, gradient,
    speed and cost matrix between all vertices. Costs are CO2 emissions
    along each path and can be computed using either MEET or COPERT methodologies.
    """
    def __init__(self, N, h, load=True):

        # Base grid defined as self.xx and self.yy
        self.x, self.y = np.arange(0, N, h), np.arange(0, N, h)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        # Placeholders made for elevation and vertex values
        # self.elevation and self.vertice are NxN matrices
        # following the shape of the grid
        self.elevation = np.zeros((N, N))
        self.vertice = np.zeros((N, N), dtype=int)

        # Call in coefficients for both methodologies
        self.MEETdf = pd.read_csv("data\MEET_Slope_Correction_Coefficients_Light_Diesel_CO2.csv")
        self.COPERTdf = pd.read_csv("data\Copert_GVRP_Coefficients.csv").iloc[4:]

        # Load correction factor for MEET model
        self.load = load

        # Convert grid coordinates to array of (x, y) points
        self.points = np.append(self.yy.reshape(-1,1), self.xx.reshape(-1,1), axis=1)


    def add_elevation_points(self, data):
        """ Add elevation values to points as described by input data

        data - numpy array of points, points have shape (x, y, elevation)
        """

        # Loop through each input datapoint and add it's
        # elevation to the corresponding gridpoint
        for i in data:
            self.elevation[i[1], i[0]] = i[2]


    def add_vertices(self, data):
        """ Define specific points as vertices as described by input data

        data - numpy array of points, points have shape (x, y)
        """

        # Loop through each input datapoint and add it
        # as a vertex to grid with a unique index
        self.vertice_count = 1
        for i in data:
            self.vertice[i[0], i[1]] = self.vertice_count
            self.vertice_count += 1


    def create_interpolation(self, data):
        """ Take in input data and interpolate elevations over entire grid.
            Interpolation carried out with IDW2 method.

        data - numpy array of points, points have shape (x, y, elevation)
        """

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
        self.elevation = np.reshape(weighted_averages, (len(self.xx), len(self.yy)))

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
            # Find Euclidean distance between all vertices
            # uses scipy.spatial.distance_matrix
            self.distance_matrix = pd.DataFrame(sc.spatial.distance_matrix(data[:,0:2], data[:,0:2]),
                                                index=data[:, 3].astype(int), columns=data[:, 3].astype(int))

        else:
            print("Choose valid method")


    def compute_gradient(self):
        """ Compute gradients between all vertices. Represented as pandas DataFrame
            Gradient defined as Rise/Run in %
        """

        # Identify all vertices in grid
        data = self.df[self.df['is_vertice'] != 0].values

        # Empty DataFrame to insert values
        self.gradient_matrix = pd.DataFrame(np.zeros((len(data), len(data))),
                                            index=data[:, 3].astype(int), columns=data[:, 3].astype(int))

        # Loop through each edge in DataFrame
        for i in data:
            for j in data:

                # Index of edge
                index = int(i[3]), int(j[3])

                # Compute rise, run accessed from distance matrix
                rise = i[2] - j[2]
                run = self.distance_matrix[index[0]][index[1]]

                # Add value to matrix (without counting diagonals)
                if run != 0:
                    self.gradient_matrix[index[0]][index[1]] = rise/run


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

        # Identify all vertices in grid
        data = self.df[self.df['is_vertice'] != 0].values

        # Empty DataFrame to insert values
        self.speed_matrix = pd.DataFrame(np.zeros((len(data), len(data))),
                                         index=data[:, 3].astype(int), columns=data[:, 3].astype(int))

        # Average velocities from driving cycle
        velocity = self.dc_net['v_ave with stops (km/h)']

        # Loop through each edge in DataFrame
        for i in data:
            for j in data:

                # Find distance of path
                d = self.distance_matrix[i[3]][j[3]]

                # Assign average speed travelled along path according to separation
                if d < 10:
                    self.speed_matrix[i[3]][j[3]] = velocity[0]
                elif d < 20:
                    self.speed_matrix[i[3]][j[3]] = velocity[1]
                elif d < 50:
                    self.speed_matrix[i[3]][j[3]] = velocity[2]
                elif d >= 50:
                    self.speed_matrix[i[3]][j[3]] = velocity[3]


    def compute_cost(self, method="MEET"):
        """ Compute CO2 emitted along paths between all vertices using either
            MEET or COPERT methodologies. Represented as pandas DataFrame.
            Model coefficients from MEET or COPERT files previously read.

            method - input method for emissions model
        """

            # Identify all vertices in grid
            data = self.df[self.df['is_vertice'] != 0].values

            # Empty DataFrame to insert values
            self.cost_matrix = pd.DataFrame(np.zeros((len(data), len(data))),
                                            index=data[:, 3].astype(int), columns=data[:, 3].astype(int))

            # Available gradients in both models
            grads = [-6, -4, -2, 0, 2, 4, 6]

            # Using MEET model
            if method.upper() == "MEET":

                # Loop through each edge in DataFrame
                for i in data:
                    for j in data:

                        # Distance and average velocity along path
                        d = self.distance_matrix[i[3]][j[3]]
                        v = self.speed_matrix[i[3]][j[3]]

                        # Gradient along path, rounded to nearest available option
                        grad = min(grads, key=lambda x:abs(x-(self.gradient_matrix[i[3]][j[3]]) * 100))

                        # Base emisisons factor (EF) in g/km
                        EF = (429.51 - 7.8227*v + 0.0617*(v**2))

                        # Gradient correction factor (multiplicative)
                        if grad != 0:

                            # Read coefficients from MEET DataFrame
                            cf = self.MEETdf[self.MEETdf['Slope (%)'] == grad].loc[:, "A6":"A0"].values[0, :]
                            EF = (cf[0]*v**6 + cf[1]*v**5 + cf[2]*v**4 + cf[3]*v**3 + cf[4]*v**2 + cf[5]*v + cf[6]) * EF

                        # Load correction factor (multiplicative)
                        if self.load:
                            EF = ((1.27) + (0.0614*grad) + (-0.0011*grad**2) + (-0.00235*v) + (-1.33/v)) * EF

                        # Total CO2 emitted is emissions factor times distance travelled
                        cost = EF * d

                        # Add to DataFrame rounded to nearest gram
                        self.cost_matrix[i[3]][j[3]] = round(cost)

            # Using COPERT model
            elif method.upper() == "COPERT":

                # Loop through each edge in DataFrame
                for i in data:
                    for j in data:

                        # Distance and average velocity along path
                        d = self.distance_matrix[i[3]][j[3]]
                        v = self.speed_matrix[i[3]][j[3]]

                        # Gradient along path, rounded to nearest available option
                        grad = min(grads, key=lambda x:abs(x-(self.gradient_matrix[i[3]][j[3]]) * 100))

                        # Read coefficients according to gradient (assuming 50% capacity)
                        cf = self.COPERTdf[(self.COPERTdf['Road.Slope'] == grad/100) & (self.COPERTdf['Load'] == 0.5)].loc[:, "Alpha":"Hta"].values[0, :]

                        # Calculate emissions factor (energy consumed in MJ/km)
                        EF = (cf[0]*v**2 + cf[1]*v + cf[2] + cf[3]/v) / (cf[4]*v**2 + cf[5]*v + cf[6])

                        # Convert to g/km from tables in Ntziachristos COPERT report
                        EF = (EF/4.31) * 101 * 3.169

                        # Total CO2 emitted is emissions factor times distance travelled
                        cost = EF * d

                        # Add to DataFrame rounded to nearest gram
                        self.cost_matrix[i[3]][j[3]] = round(cost)

            # TODO: add FMEFCM idling rates to cost computation
            else:
                print("Choose valid method")
