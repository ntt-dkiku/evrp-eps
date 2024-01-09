import argparse
import datetime
import json
import os
import torch
import torch.optim as optim
from tqdm import tqdm
from models.am import AM4CIRP
import models.baselines as rl_baseline
from utils.util import set_device, fix_seed
from generate_dataset import CIRPDataset

def main(args: argparse.Namespace) -> None:
    # save parameter settings
    if os.path.exists(args.checkpoint_dir):
        response = input(f"The directory '{args.checkpoint_dir}' already exists. Do you want to overwrite it? [y/n]: ").strip().lower()
        if response != 'y':
            assert False, "If you don't want to overwrite the checkpoint directory, please specify another checkpoint_dir."
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(f'{args.checkpoint_dir}/cmd_args.dat', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # set random seed
    fix_seed(args.random_seed)
    
    # device settings (gpu or cpu)
    use_cuda, device = set_device(args.gpu)

    # dataset
    dataset = CIRPDataset().load_from_pkl(args.dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers)

    # model & optimizer
    model = AM4CIRP(loc_dim=args.loc_dim,
                    depot_dim=args.depot_dim, 
                    vehicle_dim=args.vehicle_dim,
                    emb_dim=args.emb_dim,
                    num_heads=args.num_heads,
                    num_enc_layers=args.num_enc_layers,
                    dropout=args.dropout,
                    device=device)
    if use_cuda:
        model.to(device)
    model_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # baseline for the REINFOCE
    if args.baseline == "rollout":
        baseline = rl_baseline.RolloutBaseline(model, dataset.opts, args, device=device)
    elif args.baseline == "exponential":
        baseline = rl_baseline.ExponentialBaseline(args.beta, device=device)
    else:
        raise TypeError("Invalid baseline type :(")

    # train
    model.train()
    for epoch in range(args.epochs+1):
        # save the current checkpoint
        if epoch % args.checkpoint_interval == 0:
            print(f"Epoch {epoch}: saving a model to {args.checkpoint_dir}/model_epoch{epoch}.pth...", end="", flush=True)
            torch.save(model.cpu().state_dict(), f"{args.checkpoint_dir}/model_epoch{epoch}.pth")
            model.to(device)
            print("done.")

        with tqdm(dataloader) as tq:
            for batch_id, batch in enumerate(tq):
                if use_cuda:
                    batch = {key: value.to(device) for key, value in batch.items()}
                # add options
                batch.update({
                    "time_horizon":  args.time_horizon,
                    "vehicle_speed": args.vehicle_speed,
                    "wait_time":     args.wait_time
                })

                # output tours
                if args.debug:
                    rewards, logprobs, reward_dict, tours = model(batch, "sampling", fname=f"{batch_id}")
                    print(f"tour_length: {reward_dict['tour_length'].mean().item()}")
                    print(f"penalty: {reward_dict['penalty'].mean().item()}")
                    print(f"tour: {tours[0][0]}")
                else:
                    rewards, logprobs = model(batch, "sampling")

                # calc baseline
                baseline_v, _ = baseline.eval(batch, rewards)

                # calc loss
                advantage = (rewards - baseline_v).detach() # [batch_size]
                loss = (advantage * logprobs).mean() # batch-wise mean [1]
                
                # backprop
                model_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm, norm_type=2)
                model_optimizer.step()
                tq.set_postfix(cost=rewards.mean().item())

        # logging
        # if epoch % args.log_interval == 0:
        #     print()

        baseline.epoch_callback(model, epoch)


if __name__ == "__main__":
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    #------------------
    # general settings
    #------------------
    parser.add_argument("--random_seed",    type=int, default=1234)
    parser.add_argument("--gpu",            type=int, default=-1)
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default=f"checkpoints/model_{now.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--debug",          action="store_true")

    #------------------
    # dataset settings
    #------------------
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--problem",      type=str, default="cirp")
    parser.add_argument("--coord_dim",    type=int, default=2)
    parser.add_argument("--num_samples",  type=int, default=1)
    parser.add_argument("--num_depots",   type=int, default=1)
    parser.add_argument("--num_locs",     type=int, default=20)
    parser.add_argument("--num_vehicles", type=int, default=3)
    parser.add_argument("--vehicle_cap",  type=int, default=10)

    #-------------------
    # training settings
    #-------------------
    parser.add_argument("--batch_size",          type=int,   default=128)
    parser.add_argument("--epochs",              type=int,   default=100)
    parser.add_argument("--log_interval",        type=int,   default=20)
    parser.add_argument("--checkpoint_interval", type=int,   default=1)
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm",      type=float, default=1.0)
    parser.add_argument("--dropout",             type=float, default=0.2)
    # for greedy baseline
    parser.add_argument("--num_greedy_samples",  type=int,   default=1280)
    parser.add_argument("--greedy_batch_size",   type=int,   default=128)

    #----------------
    # model settings
    #----------------
    parser.add_argument("--loc_dim",        type=int, default=7)
    parser.add_argument("--depot_dim",      type=int, default=4)
    parser.add_argument("--vehicle_dim",    type=int, default=11)
    parser.add_argument("--emb_dim",        type=int, default=128)
    parser.add_argument("--num_heads",      type=int, default=8)
    parser.add_argument("--num_enc_layers", type=int, default=2)

    #-------------------
    # baseline settings
    #-------------------
    parser.add_argument("--baseline", type=str,   default="rollout")
    parser.add_argument("--bl_alpha", type=float, default=0.05)
    parser.add_argument("--beta",     type=float, default=0.8)
    
    #------------------
    # other parameters
    #------------------
    parser.add_argument("--vehicle_speed", type=float, default=41.0)
    parser.add_argument("--wait_time",     type=float, default=0.5)
    parser.add_argument("--time_horizon",  type=float, default=12.0)

    args = parser.parse_args()

    main(args)