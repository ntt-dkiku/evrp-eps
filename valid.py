import os
import subprocess
import argparse
from utils.util import set_device
from eval import eval

def valid(args: argparse.Namespace) -> None:
    # compare each epoch on validation datasets
    best_epoch = 0
    min_cost   = 1e+9 # a large value
    for epoch in range(args.max_epoch):
        print(f"Evaluating the model at epoch{epoch} (currently best epoch is {best_epoch}: cost={min_cost})", flush=True)
        # load a trained model
        model_path = f"{args.model_dir}/model_epoch{epoch}.pth"
        res = eval(dataset_path=args.dataset_path,
                   eval_batch_size=args.eval_batch_size,
                   model_type="rl",
                   model_path=model_path,
                   decode_type="greedy",
                   penalty_coef=args.penalty_coef,
                   vehicle_speed=args.vehicle_speed,
                   wait_time=args.wait_time,
                   time_horizon=args.time_horizon,
                   random_seed=1234, # dummy seed. we here use greedy decode, so random seed will not affect the results.
                   gpu=args.gpu,
                   num_workers=args.num_workers)
        cost = res["avg_obj"]

        if min_cost > cost:
            best_epoch = epoch
            min_cost   = cost

    # save the best epoch
    model_path = f"{args.model_dir}/model_epoch{best_epoch}.pth"
    save_path  = f"{args.model_dir}/model_bestepoch.pth"
    subprocess.run(f"cp {model_path} {save_path}")


if __name__ == "__main__":
    import datetime
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument("--gpu",              type=int, default=-1)
    parser.add_argument("--num_workers",      type=int, default=4)
    parser.add_argument("--output_dir",       type=str, default=f"results/results_{now.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--log_fname",        type=str, default=None)

    # dataset settings
    parser.add_argument("--dataset_path",    type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=256)

    # model settings
    parser.add_argument("--model_dir",    type=str,   required=True)
    parser.add_argument("--penalty_coef", type=float, default=100)
    parser.add_argument("--max_epoch",    type=int,   default=100)

    # other parameters
    parser.add_argument("--vehicle_speed", type=float, default=41.0)
    parser.add_argument("--wait_time",     type=float, default=0.5)
    parser.add_argument("--time_horizon",  type=float, default=12.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_fname is not None:
        os.makedirs(os.path.dirname(args.log_fname), exist_ok=True)
    valid(args)