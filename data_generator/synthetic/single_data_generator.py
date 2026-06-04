import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
import torch
import pickle
import argparse

def generate_hcvrp_data(hcvrp_size, veh_num):
    data = []
    seed = 24610  # Choose the seed you want to generate data for
    rnd = np.random.RandomState(seed)

    loc = rnd.uniform(0, 1, size=(hcvrp_size + 1, 2))
    depot = loc[-1]
    cust = loc[:-1]
    d = rnd.randint(1, 10, hcvrp_size + 1)
    d = d[:-1]  # The demand of the depot is 0, which does not need to be generated here

    if veh_num == 3:
        cap = [20., 25., 30.]
        thedata = [
            depot.tolist(),
            cust.tolist(),
            d.tolist(),
            np.full((3), cap).tolist()
        ]
        data.append(thedata)

    elif veh_num == 5:
        cap = [20., 25., 30., 35., 40.]
        thedata = [
            depot.tolist(),
            cust.tolist(),
            d.tolist(),
            np.full((5), cap).tolist()
        ]
        data.append(thedata)

    return np.array(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--veh_num", type=int, default=3, help="number of the vehicles; 3 or 5")
    parser.add_argument('--graph_size', type=int, default=40,
                        help="Sizes of problem instances: {40, 60, 80, 100, 120} for 3 vehicles, "
                             "{80, 100, 120, 140, 160} for 5 vehicles")

    opts = parser.parse_args()
    data_dir = 'data'
    problem = 'hcvrp'
    datadir = os.path.join(data_dir, problem)
    os.makedirs(datadir, exist_ok=True)
    print(opts.graph_size)
    filename = os.path.join(datadir, '{}_v{}_{}.pkl'.format(problem, opts.veh_num, opts.graph_size))

    dataset = generate_hcvrp_data(opts.graph_size, opts.veh_num)
    print(dataset[0])
    save_dataset(dataset, filename)

