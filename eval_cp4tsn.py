import argparse
import torch
from tqdm import tqdm
from models.tsn.tsn import CP4TSN
from models.tsn.clustered_tsn import CP4ClusteredTSN
from generate_dataset import CIRPDataset

def main(args):
    if args.clustering:
        cp4tsn = CP4ClusteredTSN(num_clusters=args.num_clusters,
                                 cluster_type=args.cluster_type,
                                 parallel=args.parallel,
                                 num_cpus=args.num_cpus,
                                 time_horizon=args.time_horizon,
                                 dt=args.dt,
                                 vehicle_speed=args.vehicle_speed,
                                 loss_coef=args.loss_coef,
                                 loc_pre_time=args.loc_pre_time,
                                 loc_post_time=args.loc_post_time,
                                 depot_pre_time=args.depot_pre_time,
                                 depot_post_time=args.depot_post_time,
                                 ensure_minimum_charge=args.ensure_minimum_charge,
                                 ensure_minimum_supply=args.ensure_minimum_supply,
                                 random_seed=args.random_seed,
                                 num_search_workers=args.num_search_workers,
                                 log_search_progress=args.log_search_progress,
                                 limit_type=args.limit_type,
                                 time_limit=args.time_limit,
                                 solution_limit=args.solution_limit)
    else:
        cp4tsn = CP4TSN(time_horizon=args.time_horizon,
                        dt=args.dt,
                        vehicle_speed=args.vehicle_speed,
                        loss_coef=args.loss_coef,
                        loc_pre_time=args.loc_pre_time,
                        loc_post_time=args.loc_post_time,
                        depot_pre_time=args.depot_pre_time,
                        depot_post_time=args.depot_post_time,
                        ensure_minimum_charge=args.ensure_minimum_charge,
                        ensure_minimum_supply=args.ensure_minimum_supply,
                        random_seed=args.random_seed,
                        num_search_workers=args.num_search_workers,
                        log_search_progress=args.log_search_progress,
                        limit_type=args.limit_type,
                        time_limit=args.time_limit,
                        solution_limit=args.solution_limit)

    dataset = CIRPDataset().load_from_pkl(args.dataset_path, load_dataopts=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=None,
                                             shuffle=False,
                                             num_workers=args.num_workers)
    
    for batch_id, batch in enumerate(tqdm(dataloader)):
        status = cp4tsn.solve(batch, log_fname=args.log_fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--num_search_workers", type=int, default=4)
    parser.add_argument("--log_search_progress", action="store_true")
    parser.add_argument("--log_fname")

    # dataset settings
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)

    # environment settings
    parser.add_argument("--time_horizon", type=int, default=12)
    parser.add_argument("--dt", type=float, default=1.)
    parser.add_argument("--vehicle_speed", type=float, default=41.)
    parser.add_argument("--loss_coef", type=int, default=100)
    parser.add_argument("--loc_pre_time", type=float, default=0.5)
    parser.add_argument("--loc_post_time", type=float, default=0.5)
    parser.add_argument("--depot_pre_time", type=float, default=0.17)
    parser.add_argument("--depot_post_time", type=float, default=0.17)
    parser.add_argument("--ensure_minimum_charge", action="store_true")
    parser.add_argument("--ensure_minimum_supply", action="store_true")
    parser.add_argument("--limit_type", type=str, default=None)
    parser.add_argument("--time_limit", type=float, default=60.)
    parser.add_argument("--solution_limit", type=int, default=100)

    # model settings
    parser.add_argument("--clustering", action="store_true")
    parser.add_argument("--cluster_type", type=str, default="kmeans")
    parser.add_argument("--num_clusters", type=int, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num_cpus", type=int, default=4)
    args = parser.parse_args()
    main(args)