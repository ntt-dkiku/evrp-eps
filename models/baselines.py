import torch
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
from scipy.stats import ttest_rel
from generate_dataset import CIRPDataset

class Baseline(object):
    """
    super class for baselines of policy gradient
    """
    def __init__(self, device):
        self.device = device

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class NoBaseline(Baseline):
    def eval(self, x, c):
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):
    def __init__(self, beta, device=None):
        super().__init__(device)
        self.beta = beta
        self.v = None

    def eval(self, x, c):
        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()
        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']


class RolloutBaseline(Baseline):
    def __init__(self, model, opts, args, epoch=0, device="cpu"):
        super().__init__(device)
        self.opts = opts
        self.num_greedy_samples = args.num_greedy_samples
        self.greedy_batch_size = args.greedy_batch_size
        self.wait_time = args.wait_time
        self.vehicle_speed = args.vehicle_speed
        self.time_horizon = args.time_horizon
        self.bl_alpha = args.bl_alpha
        self._update_model(model, epoch)

    def epoch_callback(self, model, epoch):
        """
        update current baseline policy and evaluation dataset if the policy is improved.
        
        :param model: current training policy
        :param epoch: current epoch
        """
        print("evaluating current policy on evaluation dataset")
        candidate_vals = self.rollout(model, self.dataset, self.opts).cpu().numpy() # [test_size]
        candidate_mean = candidate_vals.mean() # [1]

        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        
        # if the policy is improved
        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch)

    def rollout(self, model, dataset, opts):
        """
        compute cost with greedy decoding
        
        :param model: current training policy or baseline policy
        :param dataset: evaluation dataset for comparing current policy with baseline one
        :param opts: parsed arguments
        :return: cost of each batch [val_size]
        """
        model.eval()
        def eval_model_bat(bat):
            with torch.no_grad():
                bat = {key: value.to(self.device) for key, value in bat.items()}
                bat.update({
                    "time_horizon": self.time_horizon,
                    "vehicle_speed": self.vehicle_speed,
                    "wait_time": self.wait_time
                })
                cost, _ = model(bat, "greedy") # [batch_size x 1] 
            return cost.data.cpu()

        return torch.cat([
            eval_model_bat(bat)
            for bat in tqdm(DataLoader(dataset, batch_size=self.greedy_batch_size))
        ], 0)

    def eval(self, x, c=None):
        """
        compute cost with greedy decoding. note:self.decode_type is set to greedy here.
        
        :param x: coordinates of points [batch_size x seq_length x node_dim]
        :retrun v: baseline (value) obtained from baseline policy
        """
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x, "greedy") # greedy decoding
        # There is no loss
        return v, 0
    
    def _update_model(self, model, epoch):
        """
        update baseline policy & evaluation dataset only when current policy is stronger than baseline one.
        
        :param model: current policy 
        :param epoch: current epoch
        """
        random_seed = self.opts.random_seed + sum(self.opts.num_samples) + 1 + epoch*self.num_greedy_samples
        # replace baseline policy with current training policy
        self.model = copy.deepcopy(model)
        
        # update dataset for comparing current policy with the baseline policy
        self.dataset = CIRPDataset().generate(num_samples=self.num_greedy_samples,
                                              num_locs=self.opts.num_locs,
                                              num_depots=self.opts.num_depots,
                                              num_vehicles=self.opts.num_vehicles,
                                              vehicle_cap=self.opts.vehicle_cap,
                                              vehicle_discharge_rate=self.opts.vehicle_discharge_rate,
                                              depot_discharge_rate=self.opts.depot_discharge_rate,
                                              discharge_lim_ratio=self.opts.discharge_lim_ratio,
                                              cap_ratio=self.opts.cap_ratio,
                                              grid_scale=self.opts.grid_scale,
                                              random_seed=random_seed)

        # compute cost of updated baseline policy with greedy decoding, and store it
        self.bl_vals = self.rollout(self.model, self.dataset, self.opts).cpu().numpy() # [test_size]
        self.mean = self.bl_vals.mean() # scalar: [1]
        self.epoch = epoch

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }