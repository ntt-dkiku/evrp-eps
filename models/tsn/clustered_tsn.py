import time
import json
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
from models.tsn.tsn import CP4TSN, LARGE_VALUE
from multiprocessing import Pool
import math
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ortools.sat.python import cp_model

class ClusteringBase(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        """
        Parameters
        ----------
        input: dict of torch.tensor

        Returns
        -------
        div_input: list of dict of torch.tensor
        """
        num_locs = len(input["loc_coords"])
        veh_init_postion_id = input["vehicle_initial_position_id"]
        num_vehicles = len(veh_init_postion_id)
        num_clusters = num_vehicles
        unique_pos_ids = torch.unique(veh_init_postion_id)

        # clustering
        div_loc_id, div_depot_id = self.clustering(input, num_clusters, unique_pos_ids)

        # merge clusters that have the same initial depot (veh_init_position_id)
        div_loc_id2 = []
        div_depot_id2 = []
        div_veh_id = []
        for unique_pos_id in unique_pos_ids:
            merge_indicies = torch.where(veh_init_postion_id == unique_pos_id)[0]
            div_loc_id2.append(torch.cat([div_loc_id[merge_idx.item()] for merge_idx in merge_indicies]))
            merge_depot_id_list = [div_depot_id[merge_idx.item()] for merge_idx in merge_indicies if div_depot_id[merge_idx.item()].size()[0] > 0]
            if len(merge_depot_id_list) > 0:
                div_depot_id2.append(torch.cat([torch.cat(merge_depot_id_list), unique_pos_id[None]]))
            else:
                div_depot_id2.append(merge_depot_id_list[0])
            div_veh_id.append(merge_indicies)

        # split inputs
        div_inputs = []
        for i in range(len(unique_pos_ids)):
            div_input = {}
            for key, value in input.items():
                if "loc" in key:
                    div_input[key] = value[div_loc_id2[i]]
                elif "depot" in key:
                    div_input[key] = value[div_depot_id2[i] - num_locs]
                elif "vehicle" in key:
                    if key == "vehicle_initial_position_id":
                        cond = torch.full((len(div_depot_id2[i]),), False)
                        depot_ids = value[div_veh_id[i]]
                        for depot_id in depot_ids:
                            cond |= (div_depot_id2[i] == depot_id)
                        offst_depot_ids = torch.where(cond)[0]
                        div_input[key] = offst_depot_ids + len(div_loc_id2[i])
                    else:
                        div_input[key] = value[div_veh_id[i]]
                else:
                    div_input[key] = value
            div_inputs.append(div_input)

        # print summary
        print(f"The original problem was split into {len(div_inputs)} sub-problem(s)!")
        return div_inputs, div_loc_id2, div_depot_id2, div_veh_id

    def clustering(self, num_clusters, unique_pos_ids):
        NotImplementedError

class KmeansClustering(ClusteringBase):
    def __init__(self, num_clusters): #, random_seed, balancing=True):
        super().__init__()
        self.num_clusters = num_clusters
        # self.random_seed = random_seed

    def clustering(self, input, num_clusters, unique_pos_ids):
        # clustering locs
        loc_coords = input["loc_coords"].cpu().detach().numpy().copy()
        num_locs = len(loc_coords)
        loc_cluster_ids, loc_cluster_centers = self.balanced_kmeans(loc_coords, num_clusters)

        # clustering depots
        depot_coords = input["depot_coords"] # .cpu().detach().numpy().copy()
        depot_id = torch.arange(len(depot_coords))
        remove_mask = torch.isin(depot_id, unique_pos_ids-num_locs) # [num_depots]
        clustered_depot_id = depot_id[~remove_mask]
        clustered_depot_coords = depot_coords[~remove_mask].cpu().detach().numpy().copy()
        depot_cluster_ids, depot_cluster_centers = self.balanced_kmeans(clustered_depot_coords, num_clusters)
        
        # matching
        veh_init_pos_ids = input["vehicle_initial_position_id"]
        veh_init_pos = depot_coords[veh_init_pos_ids-num_locs].cpu().detach().numpy().copy() # [num_clusters(vehicles), dim]
        # match the inital charge station with a cluster of locs
        loc_pairs = self.cluster_matching(veh_init_pos, loc_cluster_centers)
        # match the inital charge station with a cluster of locs
        depot_pairs = self.cluster_matching(veh_init_pos, depot_cluster_centers)
        
        div_loc_id = []; div_depot_id = []
        for cluster_id in range(num_clusters):
            div_loc_id.append(torch.tensor(loc_cluster_ids[loc_pairs[cluster_id][1]]))
            div_depot_id.append(clustered_depot_id[depot_cluster_ids[depot_pairs[cluster_id][1]]] + num_locs)
        return div_loc_id, div_depot_id

    def balanced_kmeans(self, x, num_clusters, max_iters=1000):
        """
        Ref: https://stackoverflow.com/questions/5452576/k-means-algorithm-variation-with-equal-cluster-size

        Parameters
        ----------
        x: torch.tensor [num_points, dim]
        num_clusters: int
        max_iters: int

        Returns
        -------
        cluster_ids: list of torch.tensor [num_clusters, num_clustered_points]
        cluster_centers: np.array [num_clusters, dim]
        """
        cluster_size = math.ceil(len(x) / num_clusters)
        kmeans = KMeans(num_clusters, n_init=10)
        kmeans.fit(x)
        centers = kmeans.cluster_centers_
        who_is = np.tile(np.arange(num_clusters), (cluster_size+1))[:len(x)]
        centers_repeated = centers[who_is]
        distance_matrix = cdist(x, centers_repeated)
        X_assignments = linear_sum_assignment(distance_matrix)[1]
        cluster_ids = who_is[X_assignments]

        clustered_ids = []; cluster_centers = []
        for cluster_id in range(num_clusters):
            clustered_id = np.where(cluster_ids == cluster_id)[0].tolist()
            clustered_ids.append(clustered_id)
            cluster_centers.append(x[clustered_id].mean(0))
        return clustered_ids, np.array(cluster_centers)

    def cluster_matching(self, centers1, centers2):
        """
        Parameters
        ----------
        centers1: np.array [num_clusters, dim]
        centers2: np.array [num_clusters, dim]

        Returns
        pair_list: list of tuple (cluster1_id, cluster2_id)
        -------

        """
        assert len(centers1) == len(centers2)
        num_clusters = len(centers1)
        dist = (cdist(centers1, centers2) * LARGE_VALUE).astype(np.long)
        model = cp_model.CpModel()
        edge = [[model.NewBoolVar(f'e_{i}_{j}') for j in range(num_clusters)] for i in range(num_clusters)]
        for cluster_id in range(num_clusters):
            model.Add(sum(edge[cluster_id][j] for j in range(num_clusters)) == 1)
            model.Add(sum(edge[j][cluster_id] for j in range(num_clusters)) == 1)
        model.Minimize(sum(dist[i, j] * edge[i][j] for i in range(num_clusters) for j in range(num_clusters)))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        
        pair_list = []
        for i in range(num_clusters):
            for j in range(num_clusters):
                if solver.Value(edge[i][j]):
                    pair_list.append((i, j))
        assert len(pair_list) == num_clusters
        return pair_list

class RandomClustering(ClusteringBase):
    def __init__(self, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters

    def clustering(self, input, num_clusters, unique_pos_ids):
        # shuffle & split locs
        num_locs = len(input["loc_coords"])
        suffled_loc_id = torch.randperm(num_locs)
        div_loc_id = []
        st = 0; ed = 0
        for i in range(num_clusters):
            ed = st + (num_locs + i) // num_clusters
            div_loc_id.append(suffled_loc_id[st:ed]) 
            st = ed

        # shuffle & split depots
        div_depot_id = []
        num_depots = len(input["depot_coords"])
        shuffled_depot_id = torch.randperm(num_depots) + num_locs
        shuffled_depot_id = shuffled_depot_id[~torch.isin(shuffled_depot_id, unique_pos_ids)] # remove unique_pos_ids
        num_div_depots = len(shuffled_depot_id) # num_depots - num_clusters
        st = 0; ed = 0
        for i in range(num_clusters):
            ed = st + (num_div_depots + i) // num_clusters
            div_depot_id.append(shuffled_depot_id[st:ed])
            st = ed
        
        return div_loc_id, div_depot_id 

class CP4ClusteredTSN():
    def __init__(self,
                 num_clusters: int,
                 cluster_type: str = "kmeans",
                 parallel: bool = True,
                 num_cpus: int = 4,
                 *args, **kwargs):
        self.cp4tsn = CP4TSN(*args, **kwargs)
        self.num_division = num_clusters
        self.cluster_type = cluster_type
        if cluster_type == "kmeans":
            self.clustering = KmeansClustering(num_clusters)
        elif cluster_type == "random":
            self.clustering = RandomClustering(num_clusters)
        else:
            NotImplementedError
        self.parallel = parallel
        self.num_cpus = num_cpus

    def solve(self, input: dict, log_fname: str = None):
        """
        Parameters
        ----------
        input: dict of torch.tensor

        Returns
        -------
        route: list
        route_length: float
        down_rate: float
        objective_value: float
        calc_time: float
        """
        # parameters 
        num_locs = len(input["loc_coords"])
        num_vehicles = len(input["vehicle_cap"])

        # split problem into subproblems
        div_inputs, div_loc_id, div_depot_id, div_veh_id = self.clustering(input)

        # solve sub-problems
        start_time = time.perf_counter()
        if self.parallel:
            with Pool(self.num_cpus) as pool:
                results = list(pool.imap(self.cp4tsn.solve, div_inputs))
        else:
            results = []
            for div_input in div_inputs:
                results.append(self.cp4tsn.solve(div_input))
        calc_time = time.perf_counter() - start_time

        # merge results
        total_route_length = 0
        num_down_locs = np.zeros(self.cp4tsn.T)
        for result in results:
            total_route_length += result["total_route_length"]
            num_down_locs += np.array(result["num_down_locs"])
        num_down = np.mean(num_down_locs)
        down_rate = num_down / num_locs
        objective_value = total_route_length / num_vehicles + self.cp4tsn.loss_coef * down_rate
        # print(f"calc_time = {calc_time}")
        # print(f"route_len = {total_route_length}")
        # print(f"down_rate = {down_rate}")
        # print(f"obj. = {objective_value}")

        avg_actual_tour_length = (total_route_length / num_vehicles) * input["grid_scale"].item()

        summary = {
            "avg_actual_tour_length": avg_actual_tour_length,
            "avg_num_down": num_down,
            "avg_down": down_rate,
            "avg_obj": objective_value,
            "total_calc_time": calc_time
        }

        #
        dir_name = os.path.dirname(log_fname)
        os.makedirs(f"{dir_name}/batch0-sample0", exist_ok=True)

        # save summary
        with open(log_fname, "w") as f:
            json.dump(summary, f)

        # save history
        history = {
            "time": np.arange(self.cp4tsn.T) * self.cp4tsn.dt,
            "down_loc": num_down_locs.tolist()
        }
        with open(f"{dir_name}/batch0-sample0/history_data.pkl", "wb") as f:
            pickle.dump(history, f)

        return calc_time