import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
from utils.util import save_dataset, load_dataset
import random
import json
import argparse
import _pickle as cpickle
from multiprocessing import Pool

class CIRPDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset = []
        self.size = 0
        self.opts = None

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def generate(self,
                 num_samples: int, 
                 num_locs: int, 
                 num_depots: int, 
                 num_vehicles: int, 
                 vehicle_cap: float, 
                 vehicle_discharge_rate: float, 
                 depot_discharge_rate: list,
                 discharge_lim_ratio: float = 0.1,
                 cap_ratio: float = 0.8,
                 grid_scale: float = 100.0,
                 random_seed: int = 1234):
        """
        please specify the random_seed usually. specify nothing when generating eval dataset in the rollout baseline.

        Paramters
        ---------
        
        Returns
        -------
        """
        self.dataset = self.generate_dataset(num_samples=num_samples,
                                             num_locs=num_locs,
                                             num_depots=num_depots,
                                             num_vehicles=num_vehicles,
                                             vehicle_cap=vehicle_cap,
                                             vehicle_discharge_rate=vehicle_discharge_rate,
                                             depot_discharge_rate=depot_discharge_rate,
                                             discharge_lim_ratio=discharge_lim_ratio,
                                             cap_ratio=cap_ratio,
                                             grid_scale=grid_scale,
                                             random_seed=random_seed)
        self.size = len(self.dataset)
        return self

    def load_from_pkl(self,
                      dataset_path: str,
                      load_dataopts: bool = True,
                      max_load_size: int = -1):
        """
        Paramters
        ---------
        dataset_path: str
            path to a dataset
        load_dataopts: bool
            whether or not options(argparse) of datasets are loaded
        """
        assert os.path.splitext(dataset_path)[1] == ".pkl"
        if max_load_size > 0:
            self.dataset = load_dataset(dataset_path)[:max_load_size]
        else:
            self.dataset = load_dataset(dataset_path)
        self.size = len(self.dataset)
        if load_dataopts:
            self.opts = argparse.ArgumentParser()
            data_params_dir = os.path.split(dataset_path)[0]
            with open(f"{data_params_dir}/data_cmd_args.json", "r") as f:
                self.opts.__dict__ = json.load(f)
        return self

    def generate_instance(self,
                          num_locs: int,
                          num_depots: int, 
                          num_vehicles: int, 
                          vehicle_cap: float,
                          vehicle_discharge_rate: float,
                          depot_discharge_rate_candidates: float, 
                          discharge_lim_ratio: float = 0.1,
                          cap_ratio: float = 0.8,
                          grid_scale: float = 100.0,
                          random_seed: int = None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)

        coord_dim = 2
        num_nodes = num_locs + num_depots
        #-----------------------
        # vehicles (homogeneous)
        #-----------------------
        vehicle_cap = [vehicle_cap for _ in range(num_vehicles)] # [num_vehicle]
        vehicle_initial_position_id = torch.randint(num_locs, num_nodes, (num_vehicles, )) # [num_vehicles]
        vehicle_discharge_rate = torch.FloatTensor([vehicle_discharge_rate for _ in range(num_vehicles)])
        vehicle_consump_rate = torch.FloatTensor([0.161 * grid_scale for _ in range(num_vehicles)]) # 
        #-----------
        # locations
        #-----------
        # TODO : wide-range capacity candidates
        capacity_consump = {
            2.34: [0.6, 0.7],
            11.7: [1.1, 1.5, 1.7],
            35.1: np.arange(1.1, 6.0, 0.1).tolist(),
            46.8: np.arange(1.1, 6.0, 0.1).tolist()
        }
        weights = [6, 9, 51, 59]
        loc_coords = torch.FloatTensor(num_locs, coord_dim).uniform_(0, 1) # [num_locs x coord_dim]
        loc_cap = torch.FloatTensor(random.choices(list(map(lambda x: round(x, 2), capacity_consump.keys())), k=num_locs, weights=weights)) # [num_locs]
        loc_initial_battery = (torch.rand(num_locs) * .5 + .5) * loc_cap  # 50 - 100% of the capacity [num_locs]
        # conditional probability
        loc_consump_list = []
        for cap in loc_cap:
            loc_consump_list.append(random.choices(capacity_consump[round(cap.item(), 2)], k=1))
        loc_consump_rate = torch.FloatTensor(loc_consump_list).squeeze(1) # [num_locs]
        #--------
        # depots
        #--------
        depot_coords = torch.FloatTensor(num_depots, coord_dim).uniform_(0, 1) # [num_depots x coord_dim]
        depot_discharge_rate = torch.FloatTensor(random.choices(depot_discharge_rate_candidates, k=num_depots, weights=[0.2, 0.8])) # [num_depots]
        # ensure num. of depots whose discharge = 50 is more than 50 %
        min_depot_count = int(0.5 * len(depot_discharge_rate))
        if torch.count_nonzero(depot_discharge_rate > 10) < min_depot_count:
            idx = random.sample(range(len(depot_discharge_rate)), k=min_depot_count)
            depot_discharge_rate[idx] = 50.0
           
        return {
            "grid_scale": torch.FloatTensor([grid_scale]),
            "loc_coords": loc_coords,
            "loc_cap": loc_cap * cap_ratio,
            "loc_consump_rate": loc_consump_rate,
            "loc_initial_battery": loc_initial_battery * cap_ratio,
            "depot_coords": depot_coords,
            "depot_discharge_rate": depot_discharge_rate,
            "vehicle_cap": torch.FloatTensor(vehicle_cap) * cap_ratio,
            "vehicle_initial_position_id": vehicle_initial_position_id,
            "vehicle_discharge_rate": vehicle_discharge_rate,
            "vehicle_consump_rate": vehicle_consump_rate,
            "vehicle_discharge_lim": discharge_lim_ratio * torch.FloatTensor(vehicle_cap)
        }

    def generate_dataset(self,
                         num_samples: int, 
                         num_locs: int, 
                         num_depots: int, 
                         num_vehicles: int, 
                         vehicle_cap: float, 
                         vehicle_discharge_rate: float, 
                         depot_discharge_rate: list,
                         discharge_lim_ratio: float = 0.1,
                         cap_ratio: float = 0.8,
                         grid_scale: float = 100.0,
                         random_seed: int = 1234):
        seeds = random_seed + np.arange(num_samples)
        return [
            self.generate_instance(num_locs=num_locs,
                                   num_depots=num_depots,
                                   num_vehicles=num_vehicles,
                                   vehicle_cap=vehicle_cap,
                                   vehicle_discharge_rate=vehicle_discharge_rate,
                                   depot_discharge_rate_candidates=depot_discharge_rate, 
                                   discharge_lim_ratio=discharge_lim_ratio, 
                                   cap_ratio=cap_ratio,
                                   grid_scale=grid_scale,
                                   random_seed=seed)
            for seed in tqdm(seeds)
        ]

    def generate_dataset_para(self,
                              num_samples: int, 
                              num_locs: int, 
                              num_depots: int, 
                              num_vehicles: int, 
                              vehicle_cap: float, 
                              vehicle_discharge_rate: float, 
                              depot_discharge_rate: list,
                              discharge_lim_ratio: float = 0.1,
                              cap_ratio: float = 0.8,
                              grid_scale: float = 100.0,
                              random_seed: int = 1234,
                              num_cpus: int = 4):
        seeds = random_seed + np.arange(num_samples)
        with Pool(num_cpus) as pool:
            dataset = list(pool.starmap(self.generate_instance, tqdm([(num_locs, 
                                                                       num_depots,
                                                                       num_vehicles,
                                                                       vehicle_cap,
                                                                       vehicle_discharge_rate,
                                                                       depot_discharge_rate,
                                                                       discharge_lim_ratio,
                                                                       cap_ratio,
                                                                       grid_scale,
                                                                       seed) for seed in seeds], total=len(seeds))))
        return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--type", type=str, nargs="*", default=["all"])
    parser.add_argument("--num_samples", type=int, nargs="*", default=[1280000, 10000, 10000])
    parser.add_argument("--num_depots", type=int, default=12)
    parser.add_argument("--num_locs", type=int, default=50)
    parser.add_argument("--num_vehicles", type=int, default=12)
    parser.add_argument("--vehicle_cap", type=float, default=60.0) # all the vehicles have the same capacity
    parser.add_argument("--vehicle_discharge_rate", type=float, default=10.0)
    parser.add_argument("--depot_discharge_rate", type=float, nargs="*", default=[3.0, 50.0])
    parser.add_argument("--cap_ratio", type=float, default=0.8)
    parser.add_argument("--discharge_lim_ratio", type=float, default=0.1)
    parser.add_argument("--grid_scale", type=float, default=100.0)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num_cpus", type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # validation check
    if args.type[0] == "all":
        assert len(args.num_samples) == 3
    else:
        assert len(args.type) == len(args.num_samples)
    num_samples = np.sum(args.num_samples)

    if args.parallel:
        dataset = CIRPDataset().generate_dataset_para(num_samples=num_samples,
                                                      num_locs=args.num_locs,
                                                      num_depots=args.num_depots,
                                                      num_vehicles=args.num_vehicles,
                                                      vehicle_cap=args.vehicle_cap,
                                                      vehicle_discharge_rate=args.vehicle_discharge_rate,
                                                      depot_discharge_rate=args.depot_discharge_rate,
                                                      discharge_lim_ratio=args.discharge_lim_ratio,
                                                      cap_ratio=args.cap_ratio,
                                                      grid_scale=args.grid_scale,
                                                      random_seed=args.random_seed,
                                                      num_cpus=args.num_cpus)
    else:
        dataset = CIRPDataset().generate_dataset(num_samples=num_samples,
                                                num_locs=args.num_locs,
                                                num_depots=args.num_depots,
                                                num_vehicles=args.num_vehicles,
                                                vehicle_cap=args.vehicle_cap,
                                                vehicle_discharge_rate=args.vehicle_discharge_rate,
                                                depot_discharge_rate=args.depot_discharge_rate,
                                                discharge_lim_ratio=args.discharge_lim_ratio,
                                                cap_ratio=args.cap_ratio,
                                                grid_scale=args.grid_scale,
                                                random_seed=args.random_seed)
    if args.type[0] == "all":
        types = ["train", "valid", "eval"]
    else:
        types = args.type
    num_sample_list = args.num_samples
    num_sample_list.insert(0, 0)
    start = 0
    for i, type_name in enumerate(types):
        start += num_sample_list[i]
        end = start + num_sample_list[i+1]
        divided_datset = dataset[start:end]
        save_dataset(divided_datset, f"{args.save_dir}/{type_name}_dataset.pkl")
    
    # save paramters
    with open(f'{args.save_dir}/data_cmd_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)