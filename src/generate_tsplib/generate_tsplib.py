""" this script details a function to generate an instance file in TSPLib formatting """


def generate_tsplib(filename, instance_name, capacity, edge_weight_type, edge_weight_format, nodes, demand,
                    depot_index, edge_weights):
    """ Generate and save a .gvrp file in TSPLib format

    Keyword arguments:
    filename - destination of file to be saved, not including .gvrp extension.
    instance_name - name of instance.
    capacity - capacity of vehicles, assumed constant.
    edge_weight_type - specify how edge weights are given.
    edge_weight_format - specify format of explicit edge weights.
    nodes - array containing node coordinates. Two coordinates expected, in separate columns.
    demand - array specifying the demands of each node. Demands of depots should be 0.
    depot_index - indices specifying which nodes in "nodes" array correspond to depots.
    edge_weights - explicit edge weights, given either as a weight list or matrix.
    """
    with open(filename + ".gvrp", "w") as f:
        print("NAME: ", instance_name, file=f)
        print("COMMENT: this is testing the generating format", file=f)
        print("TYPE: GVRP", file=f)
        print("DIMENSION: ", len(nodes), file=f)
        print("CAPACITY: ", capacity, file=f)
        # note left_graph_type unspecified as all graphs complete
        print("EDGE_TYPE: DIRECTED", file=f)
        print("EDGE_WEIGHT_TYPE: ", edge_weight_type, file=f)
        print("EDGE_WEIGHT_FORMAT: ", edge_weight_format, file=f)
        # left edge_data_format unspecified as all graphs complete
        # left node_type unspecified as nodes unweighted?
        print("NODE_COORD_TYPE: TWOD_COORDS", file=f)  # 2d coords specified
        # display_data_type left unspecified - change if needed
        print("NODE_COORD_SECTION", file=f)
        for i in range(len(nodes)):
            print(i+1, nodes[i][0], nodes[i][1], file=f)
        print("DEPOT_SECTION:", file=f)
        for i in range(len(depot_index)):
            print(depot_index[i], file=f)
        print(-1, file=f)
        print("DEMAND_SECTION", file=f)
        for i in range(len(nodes)):
            print(str(i+1), str(demand[i]), file=f)
        # fixed_edges_section left unspecified
        # display_data_section left unspecified
        # node_weight_section left unspecified
        # tour_section
        # edge_data_section unspecified as graph complete?
        print("EDGE_WEIGHT_SECTION", file=f)
        # allows for two specifications
        if edge_weight_format == "WEIGHT_LIST":
            for i in range(len(edge_weights)):
                print(i+1, edge_weights[i][0], edge_weights[i][1], edge_weights[i][2], file=f)
        if edge_weight_format == "FULL_MATRIX":
            print(edge_weights, file=f)
        print("EOF", file=f)
    return
