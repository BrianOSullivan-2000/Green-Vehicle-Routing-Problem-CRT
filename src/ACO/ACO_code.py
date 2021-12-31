import random
import numpy as np
#create_from_file scans through the lines of the txt file and creates
#the objects we need for the optimisation
def create_from_file(file_path):
    node_list = []
    demand = []
    cost_mat = [[]]
    a = 0
    num_lines = sum(1 for line in open(file_path))
    with open(file_path, 'rt') as f:
        count = 1
        for line in f:
            if count == 1:
                t = line.split()
                name = t[1]
                count += 1
                continue
            if count > 1 and count < 4:
                count += 1
                continue
            if count == 4:
                t = line.split()
                node_num = int(t[1])
                count += 1
                continue
            if count == 5:
                t = line.split()
                capacity = int(t[1])
            elif count >= 11 and count <= node_num + 10:
                t = line.split()
                t[0], t[1], t[2] = int(t[0]), float(t[1]), float(t[2])
                node_list.append(t)
            elif count >= node_num + 15 and count < 2*node_num + 15:
                t = line.split()
                s = float(t[1])
                demand.append(s)
            elif count >= node_num + 116 and count < num_lines:
                if len(cost_mat[a]) == 0:
                    t = line.split()
                    s = [item.translate({ ord(c): None for c in "][" }) for item in t]
                    for item in s:
                        if len(item) == 0:
                            s.remove(item)
                    s = [float(item) for item in s]
                    cost_mat[a] = s
                elif len(cost_mat[a]) == 100:
                    t = line.split()
                    s = [item.translate({ ord(c): None for c in "][" }) for item in t]
                    for item in s:
                        if len(item) == 0:
                            s.remove(item)
                    s = [float(item) for item in s]
                    cost_mat.append(s)
                    a += 1

                elif 0 < len(cost_mat[a]) < 100:
                    t = line.split()
                    s = [item.translate({ ord(c): None for c in "][" }) for item in t]
                    for item in s:
                        if len(item) == 0:
                            s.remove(item)
                    s = [float(item) for item in s]
                    cost_mat[a] += s
            count += 1
    pheromone_mat = [[1 for i in range(node_num)] for j in range(node_num)]
    return capacity, np.array(cost_mat), demand, node_list, node_num, np.array(pheromone_mat)

#searches for the least costly edge adjacent to the current index, and appends the other
#incident node to the tour, and makes it our new current index
def NN_move_new(cost_mat, current_index, demand, index_to_visit, load, tour):
    possible_moves = [i for i in index_to_visit if cost_mat[current_index][i] != 0]
    move_costs = [(cost_mat[current_index][i], i) for i in possible_moves]
    ordered_moves = sorted(move_costs)
    nn_move = ordered_moves[0][1]
    tour.append(nn_move)
    load += demand[nn_move]
    current_index = nn_move
    return current_index, load, tour

#uses the NN_move_new function to create a complete tour of the nodes
#this is to give us an initial rough approximation of the optimal solution
def NN_run(capacity, cost_mat, demand, nodes):
    load = 0
    current_index = 0
    tour = [0]
    index_to_visit = [i for i in nodes if i not in tour]
    current_index = 0
    while len(index_to_visit) > 0:
        if len(adjacent_index(current_index, cost_mat, index_to_visit)) > 0:
            available_nodes = [i for i in adjacent_index(current_index, cost_mat, index_to_visit) if load + demand[i] <= capacity]
            if len(available_nodes) == 0:
                current_index, load, tour = return_to_depot(tour)
            elif len(available_nodes) > 0:
                current_index, load, tour = NN_move_new(cost_mat, current_index, demand, index_to_visit, load, tour)
                index_to_visit.remove(current_index)
                load += demand[current_index]
        elif len(adjacent_index(current_index, cost_mat, index_to_visit)) == 0:
            tour.append(tour[-2])
            current_index = tour[-2]
    tour.append(0)
    cost = total_cost(cost_mat, tour)
    return tour, cost

#from a probability distribution provided by the pheromone matrix and cost matrix,
#this function will choose a node index
def pick_probs(probabilities, nodes, pick_best = 0.1):
    x = np.random.random()
    if x < pick_best:
        p = np.argmax(probabilities)

    else:
        p = np.random.choice(range(len(probabilities)), p=probabilities)
    return nodes[p]

#calculate the total cost of travelling a given route
def total_cost(cost_mat, tour):
    cost = 0
    for i in range(len(tour)-1):
        cost += cost_mat[tour[i]][tour[i+1]]
    return cost

#computes the set of nodes that are connected to the current index node by an edge
#of the graph
def adjacent_index(current_index, cost_mat, nodes):
    adjacent = [i for i in nodes if cost_mat[current_index][i] != 0]
    return adjacent

#append a visit to the depot to the tour, and reset the load
def return_to_depot(tour):
    tour.append(0)
    load = 0
    current_index = 0
    return current_index, load, tour

#takes in a tour, and adds pheromone to the pheromone matrix based upon the edges
#travelled by the tour, and also causes the existing pheromone to decay
def update_pheromone(cost_mat, node_num, pheromone_mat, tour, decay = 0.8):
    pheromone_mat = [[pheromone_mat[i][j]*decay for i in range(len(pheromone_mat))] for j in range(len(pheromone_mat))]
    for i in range(len(tour[:-1])):
        pheromone_mat[tour[i]][tour[i + 1]] += 10*node_num/total_cost(cost_mat, tour)
    return pheromone_mat

#move to a node which has already been visited, stochastically (impacted by the ACO formula),
#for when no new nodes are available
def stoch_move_old(alpha, beta, current_index, tour, nodes, pheromone_mat, cost_mat):
    adj_nodes = [i for i in nodes if i in adjacent_index(current_index, cost_mat, nodes)]
    pheromone = [pheromone_mat[current_index][i] for i in adj_nodes]
    cost = [cost_mat[current_index][i] for i in adj_nodes]
    opt_probs = [(i**beta)*(1/j**alpha) for i, j in zip(pheromone, cost)]
    opt_probs = [i/(sum(opt_probs)) for i in opt_probs]
    p = pick_probs(opt_probs, adj_nodes)
    tour.append(p)
    current_index = p
    return current_index, tour

#move to a node which has not yet been visited, stochastically (impacted by the ACO formula)
def stoch_move_new(alpha, beta, current_index, tour, index_to_visit, load, pheromone_mat, cost_mat):
    adj_new_nodes = [i for i in index_to_visit if cost_mat[current_index][i] > 0]
    pheromone = [pheromone_mat[current_index][i] for i in adj_new_nodes]
    cost = [cost_mat[current_index][i] for i in adj_new_nodes]
    opt_probs = [(i**beta)*(1/j**alpha) for i, j in zip(pheromone, cost)]
    norm_opt_probs = [i/(sum(opt_probs)) for i in opt_probs]
    p = pick_probs(norm_opt_probs, adj_new_nodes)
    tour.append(p)
    current_index = p
    load += demand[current_index]
    return current_index, load, tour

#this function puts it all together, but has some extra features, such as deleting
#circuits that contribute no new nodes
best_tour = []
def run_ants(alpha, beta, capacity, cost_mat, iterations, nodes, pheromone_mat):
    global best_tour
    capacity, cost_mat, demand, node_list, node_num, pheromone_mat = create_from_file("Desktop/sample.gvrp.txt")
    tour = [0]
    load = 0
    repeated = 0
    tour_costs = []
    best_tour, best_cost = NN_run(capacity, cost_mat, demand, nodes)
    #tour_costs.append(best_cost)
    #print(total_cost(cost_mat, best_tour))
    update_pheromone(cost_mat, node_num, pheromone_mat, tour, decay = 0.4)
    for _ in range(iterations):
        tour = [0]
        load = 0
        current_index = 0
        index_to_visit = [i for i in nodes if i not in tour]
        while len(index_to_visit) > 0:
            if len(adjacent_index(current_index, cost_mat, index_to_visit)) > 0:
                available_nodes = [i for i in adjacent_index(current_index, cost_mat, index_to_visit) if load + demand[i] <= capacity]
                if len(available_nodes) == 0:
                    current_index, load, tour = return_to_depot(tour)
                    #pheromone_mat = update_pheromone(cost_mat, node_num, pheromone_mat, tour, decay = 0.9)
                elif len(available_nodes) > 0:
                    current_index, load, tour = stoch_move_new(alpha, beta, current_index, tour, index_to_visit, load, pheromone_mat, cost_mat)
                    #pheromone_mat = update_pheromone(cost_mat, node_num, pheromone_mat, tour, decay = 0.9)
                    index_to_visit.remove(current_index)
                track_past = []
            elif len(adjacent_index(current_index, cost_mat, index_to_visit)) > 0:
                if current_index in track_past[:-1]:
                    for i in range(2, len(track_past) + 1):
                        if current_index == tour[-i]:
                            tour = tour[:-i+1]
                            track_past = track_past[:-i+1]
                current_index, tour = stoch_move_old(alpha, beta, current_index, tour, nodes, pheromone_mat, cost_mat)
        tour.append(0)
        pheromone_mat = update_pheromone(cost_mat, node_num, pheromone_mat, tour, decay = 0.9)
        if total_cost(cost_mat, tour) < best_cost:
            best_tour = tour
            best_cost = total_cost(cost_mat, tour)
            repeated = 0
        elif total_cost(cost_mat, tour) == best_cost:
            repeated += 1
        tour_costs.append(total_cost(cost_mat, tour))
        if repeated == 30:
            break
        pheromone_mat = update_pheromone(cost_mat, node_num, pheromone_mat, best_tour, decay = 0.9)
    return best_cost, best_tour, tour_costs
