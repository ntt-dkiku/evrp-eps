import torch
import torch.nn as nn
import torch.nn.functional as F
from models.state import CIRPState

PENALTY_COEF = 100

class NaiveModel(nn.Module):
    def __init__(self,
                 model_type = "naive_greedy",
                 device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model_type = model_type

    def decode(self,
               inputs: dict):
        """
        Parameters
        ----------

        Returns
        -------

        """
        state, tour_list, vehicle_list, skip_list = self._rollout(inputs)
        cost_dict = state.get_rewards()
        vehicle_ids = torch.cat(vehicle_list, -1) 
        node_ids    = torch.cat(tour_list, -1)
        pad_masks   = torch.cat(skip_list, -1)
        return cost_dict, vehicle_ids, node_ids, pad_masks

    def _rollout(self, 
                 input: dict, 
                 fname: str = None):
        """
        Parameters
        ----------
        input: dict
        decode_type: str
        fname: str
        
        Returns
        -------
        state: CIRPState
        prob_list: list
        tour_list: list
        vehicle_list: list
        skip_list: list
        """
        tour_list = []; vehicle_list = []; skip_list = []
        state = CIRPState(input, self.device, fname)
        while not state.all_finished():
            loc_feats, depot_feats, vehicle_feats = state.get_inputs()
            node_mask = state.get_mask() # [batch_size x num_nodes]
            # decoding
            selected_vehicle_id = state.get_selected_vehicle_id()
            if self.model_type == "naive_greedy":
                selected_node_ids = self.greedy_decode(loc_feats, depot_feats, vehicle_feats, selected_vehicle_id, node_mask) # [batch_size]
            elif self.model_type == "naive_random":
                selected_node_ids = self.random_decode(node_mask) # [batch_size]
            elif self.model_type == "wo_move":
                selected_node_ids = self.without_move(state.get_curr_nodes())
            else:
                NotImplementedError
            # store results
            tour_list.append(selected_node_ids[:, None])
            vehicle_list.append(state.next_vehicle_id[:, None])
            skip_list.append(state.skip[:, None])
            # update state
            state.update(selected_node_ids) # [batch_size]
        return state, tour_list, vehicle_list, skip_list
    
    def greedy_decode(self,
                      loc_feats,
                      depot_feats,
                      vehicle_feats,
                      selected_vehicle_id,
                      node_mask):
        """
        Parameters
        ----------
        
        Returns
        -------
        selected_node_ids: torch.LongTensor [batch_size]
        """
        num_locs = loc_feats.size(1)
        # if at least one visitable location exists, select the location whose battery is minimum as the next destination
        loc_mask = node_mask[:, :num_locs] # [batch_size, num_locs]
        loc_batch = loc_mask.sum(-1) > 0 # [batch_size]
        loc_batt = loc_feats[:, :, -1] # [batch_size, num_locs]
        loc_batt_min_idx = (loc_batt + ~loc_mask * 1e+9).min(-1)[1] # remove unvisitable locations by adding a large value [batch_size]
        selected_node_ids = loc_batt_min_idx * loc_batch
        
        # if no visitable location exists, select the nearest depot to the current position as the next destination
        depot_batch = ~loc_batch
        batch_size = depot_batch.size(0)
        vehicle_corrds = vehicle_feats[:, :, 1:3] # [batch_size, num_vehicles, coord_dim]
        curr_coords = vehicle_corrds.gather(1, selected_vehicle_id[:, None, None].expand(batch_size, 1, vehicle_corrds.size(-1))) # [batch_size, 1, coord_dim]
        depot_coords = depot_feats[:, :, :2] # [batch_size, num_depots, coord_dim]
        nearest_depot_idx = torch.min(torch.linalg.norm(curr_coords - depot_coords, dim=-1), -1)[1] # [batch_size]
        selected_node_ids += (nearest_depot_idx + num_locs) * depot_batch # [batch_size]

        return selected_node_ids
    
    def random_decode(self,
                      node_mask):
        """
        Parameters
        ----------
        node_mask: torch.BoolTensor [batch_size, num_nodes]
        """
        num_avail_nodes = node_mask.sum(-1) # [batch_size]
        probs = (1. / num_avail_nodes).unsqueeze(-1) * node_mask
        selected = probs.multinomial(1).squeeze(1)
        # check if sampling went OK, can go wrong due to bug on GPU:
        #   points with zero probability are sampled occasionally
        # see https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
        while (node_mask.gather(1, selected.unsqueeze(-1)) == 0).data.any():
            print('Sampled bad values, resampling!')
            selected = probs.multinomial(1).squeeze(1)

        return selected

    def without_move(self,
                     curr_nodes):
        """
        Parameters
        ----------
        
        """
        return curr_nodes
    
    def replicate_batch(self,
                        inputs: dict,
                        num_replicas: int):
        """
        Replicates inputs for sampling/beam-search decoding
        
        Parameters
        ----------
        inputs: dict of torch.Tensor [batch_size x ...]
        num_replicas: int
        
        Returns
        -------
        replicated_inputs: dict of torch.Tensor [(num_replicas * batch_size) x ...]
        """
        return {
            k: v.unsqueeze(0).expand(num_replicas, *v.size()).reshape(-1, *v.size()[1:]) 
            if k not in ["time_horizon", "vehicle_speed", "wait_time"] else v
            for k, v in inputs.items()
        }

    def sample_decode(self, 
                      inputs: dict,
                      search_width: int,
                      max_batch_size: int = 1028):
        """
        Parameters
        ----------
        inputs: dict of torch.Tensor [batch_size x ...]
        search_width: int
        
        Returns
        -------
        min_cost: torch.Tensor [batch_size]
        vehicle_ids: torch.Tensor [batch_size x max_steps]
        node_ids: torch.Tensor [batch_size x max_steps]
        masks: torch.Tensor [batch_size x max_steps]
        """
        batch_size = len(inputs["loc_coords"])    
        if search_width * batch_size > max_batch_size:
            assert (max_batch_size % batch_size) == 0
            assert ((search_width * batch_size) % max_batch_size) == 0
            rep_batch = max_batch_size // batch_size
            num_itr = (search_width * batch_size) // max_batch_size
        else:
            rep_batch = search_width
            num_itr = 1
        penalty_coef = PENALTY_COEF
        rep_inputs = self.replicate_batch(inputs, rep_batch)
        node_id_list = []; vehicle_id_list = []; mask_list = []
        tour_length_list = []; penalty_list = []
        max_steps = 0
        for itr in range(num_itr):
            state, tour_list, vehicle_list, skip_list = self._rollout(rep_inputs)
            node_id_list.append(torch.stack(tour_list, 1).view(rep_batch, batch_size, -1))       # [rep_batch x batch_size x num_steps]    
            vehicle_id_list.append(torch.stack(vehicle_list, 1).view(rep_batch, batch_size, -1)) # [rep_batch x batch_size x num_steps]
            mask_list.append(torch.stack(skip_list, 1).view(rep_batch, batch_size, -1))          # [rep_batch x batch_size x num_steps]
            cost_dict = state.get_rewards()
            tour_length_list.append(cost_dict["tour_length"].view(rep_batch, batch_size)) # [rep_batch x batch_size]
            penalty_list.append(cost_dict["penalty"].view(rep_batch, batch_size)) # [rep_batch x batch_size]
            max_steps = max(max_steps, len(tour_list))
        # padding
        node_id_list = torch.cat([F.pad(node_ids, (0, max_steps - node_ids.size(-1)), "constant", 0) for node_ids in node_id_list], 0) # [search_width x batch_size x max_steps]
        vehicle_id_list = torch.cat([F.pad(vehicle_ids, (0, max_steps - vehicle_ids.size(-1)), "constant", 0) for vehicle_ids in vehicle_id_list], 0) # [search_width x batch_size x max_steps]
        mask_list = torch.cat([F.pad(masks, (0, max_steps - masks.size(-1)), "constant", True) for masks in mask_list], 0) # [search_width x batch_size x max_steps]
        tour_length_list = torch.cat(tour_length_list, 0) # [search_width x batch_size]
        penalty_list = torch.cat(penalty_list, 0) # [search_width x batch_size]
        cost_list = tour_length_list + penalty_coef * penalty_list # [search_width x batch_size]
        
        # extract a sample that has minimum cost
        min_cost, min_cost_idx = cost_list.min(0) # [1 x batch_size]
        min_cost_idx = min_cost_idx.reshape(1, batch_size)
        min_tour_length = tour_length_list.gather(0, min_cost_idx).squeeze(0) # [batch_size]
        min_penalty = penalty_list.gather(0, min_cost_idx).squeeze(0) # [batch_size]
        min_cost_idx = min_cost_idx.unsqueeze(-1).expand(1, batch_size, max_steps) # [1 x batch_size x max_steps]
        node_ids    = node_id_list.gather(0, min_cost_idx).squeeze(0)    # [batch_size x max_steps]
        vehicle_ids = vehicle_id_list.gather(0, min_cost_idx).squeeze(0) # [batch_size x max_steps]
        masks       = mask_list.gather(0, min_cost_idx).squeeze(0)    # [batch_size x max_steps]
        
        return {"tour_length": min_tour_length, "penalty": min_penalty}, vehicle_ids, node_ids, masks