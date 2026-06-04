from torch.utils.data import Dataset
import torch
import os
import pickle
from core.problems.hcvrp.state_hcvrp import StateHCVRP
from utils.beam_search import beam_search
import math

class HCVRP(object):
    NAME = 'hcvrp'

    VEHICLE_CAPACITY = [20., 25., 30.]

    @staticmethod # get cost modify with time window, when vehicle arrive at time 0, small reward. check from update env, traceback to attention file
    def get_costs(dataset, obj, pi, veh_list, tour_1, tour_2, tour_3):
        
        #1. without time window knowledge
        #2. with time window knowledge (need input time window data to architecture)
        
        if obj == 'min-max':
            SPEED = [1, 1, 1]
        if obj == 'min-sum':
            SPEED = [1/4, 1/5, 1/6]
        batch_size, graph_size = dataset['demand'].size()
        num_veh = len(HCVRP.VEHICLE_CAPACITY)

        # # Check that tours are valid, i.e. contain 0 to n -1, [batch_size, num_veh, tour_len]
        sorted_pi = pi.data.sort(1)[0]
        # Sorting it should give all zeros at front and then 1...n
        assert (torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
                sorted_pi[:, -graph_size:]
                ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        demand_with_depot = torch.cat(  # [batch_size, graph_size]
            (
                torch.full_like(dataset['demand'][:, :1], 0),  # pickup problem, set depot demand to -capacity
                dataset['demand']
            ),
            1
        )
        # pi: [batch_size, tour_len]
        d = demand_with_depot.gather(1, pi)
        # print(d.shape)
        # print(f"veh lis {veh_list.shape}")
        # raise ValueError
        used_cap = torch.zeros_like(dataset['demand'][:, 0:num_veh])  # batch_size, 3
        # print(f"use cap {used_cap.shape}")
        # for veh in range(num_veh):  # num_veha
        for i in range(pi.size(-1)):  # tour_len
            # print('d', i, d[0, i])
            # print(f"veh list {veh_list}")
            # raise ValueError
            used_cap[torch.arange(batch_size), veh_list[torch.arange(batch_size), i]] += d[:,
                                                                                         i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            used_cap[used_cap[torch.arange(batch_size), veh_list[torch.arange(batch_size), i]] < 0] = 0
            used_cap[(tour_1[:, i] == 0), 0] = 0
            assert (used_cap[torch.arange(batch_size), 0] <=
                    HCVRP.VEHICLE_CAPACITY[0] + 1e-5).all(), "Used more than capacity 1"
            used_cap[(tour_2[:, i] == 0), 1] = 0
            assert (used_cap[torch.arange(batch_size), 1] <=
                    HCVRP.VEHICLE_CAPACITY[1] + 1e-5).all(), "Used more than capacity 2"
            used_cap[(tour_3[:, i] == 0), 2] = 0
            assert (used_cap[torch.arange(batch_size), 2] <=
                    HCVRP.VEHICLE_CAPACITY[2] + 1e-5).all(), "Used more than capacity 3"

        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)  # batch_size, graph_size+1, 2
        
        # [batch_size, tour_len, 2]
        dis_1 = loc_with_depot.gather(1, tour_1[..., None].expand(*tour_1.size(), loc_with_depot.size(-1)))
        dis_2 = loc_with_depot.gather(1, tour_2[..., None].expand(*tour_2.size(), loc_with_depot.size(-1)))
        dis_3 = loc_with_depot.gather(1, tour_3[..., None].expand(*tour_3.size(), loc_with_depot.size(-1)))

        distance_matrices = [dis_1, dis_2, dis_3]
        tours = [tour_1, tour_2, tour_3]
        total_dists = []

        for dis, tour in zip(distance_matrices, tours):
            distance_matrix_1 = (dis[:, 1:] - dis[:, :-1]).norm(p=2, dim=2) # Batch_size x Node_size

            # distance_matrix_1 = torch.cat(((dis[:, 0, None] - dataset['depot']).norm(p = 2, dim = 1), distance_matrix_1)) # depot to node1, node1 to node2, ...
            # Calculate the norm
            norm = (dis[:, 0, None] - dataset['depot']).norm(p = 2, dim = 1)

            # Ensure the norm tensor has the same shape as distance_matrix_1 in dimension 1
            norm = norm.view(distance_matrix_1.shape[0], -1)

            # Now concatenate the tensors
            distance_matrix_1 = torch.cat((norm, distance_matrix_1), dim=1)
            time_window_1 = dataset['time_window'] # batch_size x node_size x time_window_size

            visiting_time = 0  # vehicle starts delivery at 00:00
            cost_multipliers = [] 
            time_window_resolution = 1.0

            adjsuted_dists_list = []
            for batch_idx, (dists, visited_nodes) in enumerate(zip(distance_matrix_1, tour)):
                travel_duration = dists / SPEED[0] # Batch_size;  assume distance in km, speed in km/hr
                visiting_time = visiting_time + travel_duration # Batch_size; # proses mengunjungi customer max 24 jam/index
                
                visiting_time_idxs = torch.floor(visiting_time / time_window_resolution).long() # Batch_size;
                # print(f"visiting time {visiting_time}")
                # print(f"time window resolution {time_window_resolution}")

                # print(f"visiting time idx 1 {visiting_time_idxs}")
                # print(time_window_1.size(2) - 1)
                # visiting_time_idxs = torch.clamp(visiting_time_idxs, max=time_window_1.size(2) - 1)
                visiting_time_idxs = visiting_time_idxs % 24
                # print(f"visiting time idx 1 after modulo {visiting_time_idxs}")
                # break
                # This part can be optimized!
                adjusted_dists = []
                for dist, visited_node, visiting_time_idx in zip(dists, visited_nodes, visiting_time_idxs):
                    if visited_node == 0: # visiting depot, assume depot is always open
                        # dist = dist * 0.1
                        dist = dist * 0.1
                    else:
                        visited_node = visited_node - 1 # adjust index to graph size
                        # print(type(time_window_1[batch_idx, visited_node, visiting_time_idx]))
                        if time_window_1[batch_idx, visited_node, visiting_time_idx] == 1:
                            dist = dist * 0.1 # Visiting during opening time
                        else:
                            dist = dist * 2.0 # Visiting during closing time
                    adjusted_dists.append(dist)
                adjusted_dists = torch.stack(adjusted_dists, dim=0)
                adjsuted_dists_list.append(adjusted_dists)

            adjsuted_dists_list = torch.stack(adjsuted_dists_list, dim=0)
            total_dists.append(adjsuted_dists_list)  # [batch_size]

        total_dis = torch.cat(total_dists, -1)

        if obj == 'min-max':
            return torch.max(total_dis, dim=1)[0], None
        if obj == 'min-sum':
            return torch.sum(total_dis, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return HCVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateHCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = HCVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, time_window, *args = args  # Include time_window
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float),  # Scale demand
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'capacity': torch.tensor(capacity, dtype=torch.float),
        'time_window': torch.tensor(time_window, dtype=torch.float),  # Add time_window
    }

class HCVRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=10000, offset=0, distribution=None):
        super(HCVRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: [20., 25., 30.],
                20: [20., 25., 30.],
		30: [20., 25., 30.],
                40: [20., 25., 30.],
                50: [20., 25., 30.],
                60: [20., 25., 30.],
                80: [20., 25., 30.],
                100: [20., 25., 30.],
                120: [20., 25., 30.],
            }
            # capa = torch.zeros((size, CAPACITIES[size]))

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float(),
                    'depot': torch.FloatTensor(2).uniform_(0, 1),
                    'capacity': torch.Tensor(CAPACITIES[size]),
                    'time_window': self.generate_time_windows(size)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)  # num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]  # index of sampled data
    
    @staticmethod
    def generate_time_windows(size):
        # Define shift timings
        SHIFT_1_START = 7  # 7 AM
        SHIFT_1_END = 12   # 12 PM
        SHIFT_2_START = 13 # 1 PM
        SHIFT_2_END = 18   # 6 PM

        # Initialize time windows tensor
        time_windows = torch.zeros(size, 24)

        # Set depot time windows
        time_windows[:, SHIFT_1_START:SHIFT_1_END] = 1  # Open in shift 1
        time_windows[:, SHIFT_2_START:SHIFT_2_END] = 1  # Open in shift 2

         # Add nodes with specific shift time windows
        shift_1_nodes = torch.randint(0, size, (size // 4,))  # Assuming 25% of nodes open only in shift 1
        shift_2_nodes = torch.randint(0, size, (size // 4,))  # Assuming 25% of nodes open only in shift 2

        for node in shift_1_nodes:
            time_windows[node, SHIFT_1_START:SHIFT_1_END] = 1
            time_windows[node, SHIFT_2_START:SHIFT_2_END] = 0  # Close in shift 2

        for node in shift_2_nodes:
            time_windows[node, SHIFT_1_START:SHIFT_1_END] = 0  # Close in shift 1
            time_windows[node, SHIFT_2_START:SHIFT_2_END] = 1

        return time_windows.float()

