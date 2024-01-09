import torch
import torch.nn as nn
import torch.nn.functional as F
from models.am_encoder import AMEncoder
from models.am_decoder import AMDecoder
from models.state import CIRPState

PENALTY_COEF = 100

class AM4CIRP(nn.Module):
    def __init__(self,
                 loc_dim: int,
                 depot_dim: int,
                 vehicle_dim: int,
                 emb_dim: int,
                 num_heads: int,
                 num_enc_layers: int,
                 dropout: float = 0.0,
                 device: str = "cpu"):
        super().__init__()
        self.device = device
        self.encoder = AMEncoder(loc_dim, depot_dim, vehicle_dim, emb_dim, num_heads, num_enc_layers, dropout)
        self.decoder = AMDecoder(emb_dim)

    def forward(self,
                input: dict,
                decode_type: str, 
                fname: str = None):
        """
        Parameters
        ----------
        input:
        decode_type: str
        output_tours: bool
        Returns
        -------
        rewards: dict
            tour_length: torch.tensor [batch_size]
            penalty: torch.tensor [batch_size]
        tour_logprobs: torch.tensor [batch_size]
        tours: list of list of list [batch_size x num_vehicles x num_vehicle_steps]
        """
        #-------------
        # action loop
        #-------------
        prob_list = []; tour_list = []; vehicle_list = []; skip_list = []
        state = CIRPState(input, self.device, fname)
        while not state.all_finished():
            loc_feats, depot_feats, vehicle_feats = state.get_inputs()
            mask = state.get_mask()
            # encoding
            node_context, vehicle_context = self.encoder(loc_feats, depot_feats, vehicle_feats)
            # decoding
            selected_vehicle_id = state.get_selected_vehicle_id()
            vehicle_context = torch.gather(vehicle_context, 1, selected_vehicle_id[:, None, None].expand(state.batch_size, 1, vehicle_context.size(-1)))
            probs, selected = self.decoder(node_context, vehicle_context, mask, decode_type)
            # store results
            prob_list.append(probs)
            tour_list.append(selected[:, None])
            vehicle_list.append(state.next_vehicle_id[:, None])
            skip_list.append(state.skip[:, None])
            # update state
            state.update(selected)
        #---------
        # rewards
        #---------
        # penalty_coef = PENALTY_COEF # input["time_horizon"] * input["vehicle_speed"] * state.num_vehicles * 10
        reward_dict = state.get_rewards()
        rewards = reward_dict["tour_length"] + PENALTY_COEF * reward_dict["penalty"] # [batach_size]
        
        # log probabilities
        prob_list  = torch.stack(prob_list, 1) # [batch_size x num_steps x num_nodes]
        tour_list  = torch.stack(tour_list, 1) # [batch_size x num_steps]
        tour_probs = torch.gather(prob_list, dim=2, index=tour_list).squeeze(-1) # [batch_size x num_steps]
        skip_list  = torch.stack(skip_list, 1).squeeze(-1) # [batch_size x num_steps]
        tour_logprobs = (torch.log(tour_probs + 1e-9) * ~skip_list).sum(-1) # [batch_size]
        tour_logprobs[(tour_logprobs < -1000).detach()] = 0.0 # for numerical stability
        
        # tour list for visualization TODO:
        if fname is not None:
            state.output_batt_history()
            state.output_gif()
            vehicle_list = torch.stack(vehicle_list, 1) # [batch_size x num_steps]
            skip_list = torch.stack(skip_list, 1) # [batch_size x num_steps]
            batch_size = state.batch_size
            num_steps  = skip_list.size(-1)
            tours = [[[] for _ in range(state.num_vehicles)] for __ in range(batch_size)]
            for batch in range(batch_size):
                for step in range(num_steps):
                    if not skip_list[batch, step]:
                        tours[batch][vehicle_list[batch, step]].append((tour_list[batch, step].item()))
            return rewards, tour_logprobs, reward_dict, tours
        else:
            return rewards, tour_logprobs
            
    def _rollout(self, 
                 input: dict,
                 decode_type: str, 
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
        prob_list = []; tour_list = []; vehicle_list = []; skip_list = []
        state = CIRPState(input, self.device, fname)
        while not state.all_finished():
            loc_feats, depot_feats, vehicle_feats = state.get_inputs()
            mask = state.get_mask()
            # encoding
            node_context, vehicle_context = self.encoder(loc_feats, depot_feats, vehicle_feats)
            # decoding
            selected_vehicle_id = state.get_selected_vehicle_id()
            vehicle_context = torch.gather(vehicle_context, 1, selected_vehicle_id[:, None, None].expand(state.batch_size, 1, vehicle_context.size(-1)))
            probs, selected = self.decoder(node_context, vehicle_context, mask, decode_type)
            # store results
            prob_list.append(probs)
            tour_list.append(selected[:, None])
            vehicle_list.append(state.next_vehicle_id[:, None])
            skip_list.append(state.skip[:, None])
            # update state
            state.update(selected)
        return state, prob_list, tour_list, vehicle_list, skip_list
    
    def greedy_decode(self,
                      inputs: dict):
        state, prob_list, tour_list, vehicle_list, skip_list = self._rollout(inputs, "greedy")
        cost_dict = state.get_rewards()
        vehicle_ids = torch.cat(vehicle_list, -1)
        node_ids = torch.cat(tour_list, -1)
        masks = torch.cat(skip_list, -1)
        return cost_dict, vehicle_ids, node_ids, masks
    
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
            state, prob_list, tour_list, vehicle_list, skip_list = self._rollout(rep_inputs, "sampling")
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