
import numpy as np

# Script to manually edit the instance parameters (eg comments, names, etc.)
ns = ["20", "50", "100", "200", "500", "1000"]
depots = ["centre", "corner"]
traffics = ["weekday_offpeak", "weekday_peak", "weekend_peak"]
rains = ["heavy", "mild", "low"]


for domain in ["dublin_centre", "m50", "dublin_south"]:
    for n in ns:
        for depot in depots:
            for traffic in traffics:
                for rain in rains:

                    if domain == "dublin_south" and n == "1000":
                        #print("Ignore this combo")
                        continue

                    else:

                        path = "instances/{}/{}_rainfall/{}/{}_n{}.gvrp".format(domain, rain, traffic, depot, n)
                        f = open(path)
                        data = f.readlines()

                        node_num = data[3]
                        node_num = int(node_num.split(" ")[-1][:-1])

                        data[node_num + 14] = '0 0\n'

                        demands = np.random.choice((1, 2, 3), size=node_num, p=[0.5, 0.3, 0.2])
                        demands[0] = 0
                        lines = np.empty(node_num, dtype=object)

                        for i in range(node_num):
                            line = "{} {}\n".format(i, demands[i])
                            lines[i] = line

                        data[node_num + 14: 2*node_num + 14] = lines

                        with open(path, 'w') as file:
                            file.writelines( data )
