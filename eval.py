import os
import torch
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from utils.util import set_device, output_tour_to_csv, save_dataset, fix_seed
from generate_dataset import CIRPDataset
from models.am import AM4CIRP
from models.naive_models import NaiveModel
from models.state import visualize_routes as vis_routes
from models.state import save_route_info

def eval(dataset_path: str,
         eval_batch_size: int = 256,
         max_load_size: int = -1,
         model_type: str = "rl",
         model_path: str = None,
         model_dir: str = None,
         decode_type: str = "sampling",
         search_width: int = 12800,
         max_batch_size: int = 128,
         penalty_coef: float = 100,
         vehicle_speed: float = 41,
         wait_time: float = 0.5,
         time_horizon: float = 12,
         random_seed: int = 1234,
         gpu: int = -1,
         num_workers: int = 4,
         visualize_routes: bool = False,
         output_dir: str = None) -> Dict[str, Any]:
    #-----------------
    # set random seed
    #-----------------
    fix_seed(random_seed)
    
    #------------------------------
    # device settings (gpu or cpu)
    #------------------------------
    use_cuda, device = set_device(gpu)

    #---------
    # dataset
    #---------
    dataset = CIRPDataset().load_from_pkl(dataset_path, load_dataopts=False, max_load_size=max_load_size)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=eval_batch_size,
                                             shuffle=None,
                                             num_workers=num_workers)

    #-------
    # model
    #-------
    if model_type == "rl":
        # load a trained model
        if model_path is not None: 
            model_dir = os.path.split(model_path)[0]
        elif model_dir is not None:
            model_path = f"{model_dir}/model_bestepoch.pth"
        else:
            assert False, "specify the one from model_path and model_dir :("

        params = argparse.ArgumentParser()
        with open(f"{model_dir}/cmd_args.dat", "r") as f:
            params.__dict__ = json.load(f)
        model = AM4CIRP(loc_dim=params.loc_dim,
                        depot_dim=params.depot_dim,
                        vehicle_dim=params.vehicle_dim,
                        emb_dim=params.emb_dim,
                        num_heads=params.num_heads,
                        num_enc_layers=params.num_enc_layers,
                        dropout=params.dropout,
                        device=device)
        model.load_state_dict(torch.load(model_path))
        if use_cuda:
            model.to(device)
    elif model_type in ["naive_greedy", "naive_random", "wo_move"]:
        model = NaiveModel(model_type, device)
    else:
        raise TypeError("Invalid model_type!")

    #------------
    # evaluation
    #------------
    actual_tour_length_list = []
    tour_length_list = []
    down_list = []
    num_down_list = []
    calc_time_list = []
    model.eval()
    for batch_id, batch in enumerate(tqdm(dataloader)):
        start_time = time.perf_counter()
        
        if use_cuda:
            batch = {key: value.to(device) for key, value in batch.items()}
        # add options
        batch.update({
            "time_horizon": time_horizon,
            "vehicle_speed": vehicle_speed,
            "wait_time": wait_time
        })

        # output tours
        if model_type == "rl":
            if decode_type == "greedy":
                with torch.inference_mode():
                    cost_dict, vehicle_ids, node_ids, mask = model.greedy_decode(batch)
            elif decode_type == "sampling":
                with torch.inference_mode():
                    cost_dict, vehicle_ids, node_ids, mask = model.sample_decode(batch, search_width, max_batch_size)
            else:
                NotImplementedError
        elif model_type == "naive_random":
            cost_dict, vehicle_ids, node_ids, mask = model.sample_decode(batch, search_width, max_batch_size)
        elif model_type in ["naive_greedy", "wo_move"]:
            cost_dict, vehicle_ids, node_ids, mask = model.decode(batch)
        else:
            NotImplementedError

        calc_time_list.append(time.perf_counter() - start_time)
        tour_length_list.append(cost_dict["tour_length"])
        down_list.append(cost_dict["penalty"])
        num_down_list.append(cost_dict["penalty"] * batch["loc_coords"].size(1))
        actual_tour_length_list.append(cost_dict["tour_length"] * batch["grid_scale"].squeeze(-1))
        
        #---------------
        # visualization
        #---------------
        if visualize_routes:
            os.makedirs(output_dir, exist_ok=True)
            vis_routes(vehicle_ids, node_ids, batch, f"{output_dir}/batch{batch_id}", device)
            save_route_info(batch, vehicle_ids, node_ids, mask, f"{output_dir}/batch{batch_id}")

    #------------------
    # calculation time
    #------------------
    avg_calc_time = np.mean(calc_time_list)
    std_calc_time = np.std(calc_time_list)
    total_calc_time = np.sum(calc_time_list)
    
    #-----------------
    # objective value
    #-----------------
    tour_length = torch.cat(tour_length_list, dim=0) # [eval_size]
    down = torch.cat(down_list, dim=0) # [eval_size]
    all_costs = tour_length + penalty_coef * down # [eval_size]
    avg_obj = torch.mean(all_costs).cpu().item()
    std_obj = torch.std(all_costs, unbiased=False).cpu().item()
    avg_tour_length = torch.mean(tour_length).cpu().item()
    std_tour_length = torch.std(tour_length, unbiased=False).cpu().item()
    avg_down = torch.mean(down).cpu().item()
    std_down = torch.std(down, unbiased=False).cpu().item()
    num_down = torch.cat(num_down_list, dim=0)
    avg_num_down = torch.mean(num_down).cpu().item()
    std_num_down = torch.std(num_down, unbiased=False).cpu().item()
    actual_tour_length = torch.cat(actual_tour_length_list, dim=0)
    avg_actual_tour_length = torch.mean(actual_tour_length).cpu().item()
    std_actual_tour_length = torch.std(actual_tour_length, unbiased=False).cpu().item()

    summary = {
        "avg_calc_time": avg_calc_time,
        "std_calc_time": std_calc_time,
        "avg_obj": avg_obj,
        "std_obj": std_obj,
        "avg_tour_length": avg_tour_length,
        "std_tour_length": std_tour_length,
        "avg_down": avg_down,
        "std_down": std_down,
        "total_calc_time": total_calc_time,
        "avg_actual_tour_length": avg_actual_tour_length,
        "std_actual_tour_length": std_actual_tour_length,
        "avg_num_down": avg_num_down,
        "std_num_down": std_num_down
    }

    # save log
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        log_fname = f"{output_dir}/summary.json"
        with open(log_fname, "w") as f:
            json.dump(summary, f)

    return summary

if __name__ == "__main__":
    import datetime
    import argparse
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument("--random_seed",      type=int, default=1234)
    parser.add_argument("--gpu",              type=int, default=-1)
    parser.add_argument("--num_workers",      type=int, default=4)
    parser.add_argument("--visualize_routes", action="store_true")
    parser.add_argument("--output_dir",       type=str, default=f"results/results_{now}")
    # dataset settings
    parser.add_argument("--dataset_path",    type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_load_size",   type=int, default=-1)
    # model settings
    parser.add_argument("--model_type",     type=str, default="rl")
    parser.add_argument("--model_path",     type=str, default=None)
    parser.add_argument("--model_dir",      type=str, default=None)
    parser.add_argument("--decode_type",    type=str, default="sampling")
    parser.add_argument("--search_width",   type=int, default=12800)
    parser.add_argument("--max_batch_size", type=int, default=12800)
    parser.add_argument("--penalty_coef",   type=float, default=100)
    # other parameters
    parser.add_argument("--vehicle_speed", type=float, default=41.0)
    parser.add_argument("--wait_time",     type=float, default=0.5)
    parser.add_argument("--time_horizon",  type=float, default=12.0)
    args = parser.parse_args()

    # prepare a directory
    if args.visualize_routes:
        os.makedirs(args.output_dir, exist_ok=True)

    eval(dataset_path=args.dataset_path,
         eval_batch_size=args.eval_batch_size,
         max_load_size=args.max_load_size,
         model_type=args.model_type,
         model_path=args.model_path,
         model_dir=args.model_dir,
         decode_type=args.decode_type,
         search_width=args.search_width,
         max_batch_size=args.max_batch_size,
         penalty_coef=args.penalty_coef,
         vehicle_speed=args.vehicle_speed,
         wait_time=args.wait_time,
         time_horizon=args.time_horizon,
         random_seed=args.random_seed,
         gpu=args.gpu,
         num_workers=args.num_workers,
         visualize_routes=args.visualize_routes,
         output_dir=args.output_dir)