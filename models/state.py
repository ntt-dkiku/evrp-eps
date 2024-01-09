import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import math
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patches
import copy
import subprocess

DPI = 150
SMALL_VALUE = 1e-9
BIT_SMALL_VALUE = 1e-3
SAVE_PICTURE = True
SAVE_HISTORY = True
OUTPUT_INTERVAL = 0.01
FPS = 60
UNEQUAL_INTERVAL = False

def visualize_routes2(inputs, route_list, fname, device):
    state = CIRPState(inputs, device, fname)
    count = [2 for _ in range(len(route_list))]
    while not state.all_finished():
        selected_vehicle_id = state.get_selected_vehicle_id()
        node_id = route_list[selected_vehicle_id][count[selected_vehicle_id]]
        node_ids = torch.LongTensor([node_id])
        state.update(node_ids)
        count[selected_vehicle_id] += 1
    
    if SAVE_PICTURE:
        state.output_gif()
    if SAVE_HISTORY:
        state.output_batt_history()

def visualize_routes(inputs, vehicle_ids, node_ids, fname, device):
    if vehicle_ids.dim() < 2: # if batch_size = 1
        vehicle_ids = vehicle_ids.unsqueeze(0).expand(1, node_ids.size(-1))
        node_ids = node_ids.unsqueeze(0).expand(1, node_ids.size(-1))
        
    state = CIRPState(inputs, device, fname)
    count = 0
    while not state.all_finished():
        assert (state.next_vehicle_id != vehicle_ids[:, count]).sum() == 0
        state.update(node_ids[:, count])
        count += 1
    
    if SAVE_PICTURE:
        state.output_gif()
    if SAVE_HISTORY:
        state.output_batt_history()

def save_route_info(inputs, vehicle_ids, node_ids, mask, output_dir):
    if vehicle_ids.dim() < 2: # if batch_size = 1
        vehicle_ids = vehicle_ids.unsqueeze(0).expand(1, node_ids.size(-1))
        node_ids = node_ids.unsqueeze(0).expand(1, node_ids.size(-1))
        mask = mask.unsqueeze(0).expand(1, mask.size(-1))
    sample = 0
    loc_coords = inputs["loc_coords"][sample].tolist()
    depot_coords = inputs["depot_coords"][sample].tolist()
    veh_init_pos_ids = inputs["vehicle_initial_position_id"][sample].tolist()

    ignored_depots = (inputs["depot_discharge_rate"][sample] < 10.0).tolist()
    for veh_init_pos_id in veh_init_pos_ids:
        ignored_depots[veh_init_pos_id - len(loc_coords)] = False

    vehicle_id = vehicle_ids[sample]
    node_id = node_ids[sample]
    mask = mask[sample]
    routes = [ [] for _ in range(torch.max(vehicle_id)+1)]
    assert len(vehicle_id) == len(node_id) and len(node_id) == len(mask)
    # add initial position
    for veh_id, veh_init_pos_id in enumerate(veh_init_pos_ids):
        routes[veh_id].append(veh_init_pos_id)
    # add subsequent positions
    for step, skip in enumerate(mask):
        if skip:
            break
        else:
            routes[vehicle_id[step]].append(node_id[step].item())
    route_info = {
        "loc_coords": loc_coords,
        "depot_coords": depot_coords,
        "ignored_depots": ignored_depots,
        "route": routes
    }
    os.makedirs(f"{output_dir}-sample{sample}", exist_ok=True)
    with open(f"{output_dir}-sample{sample}/route_info.pkl", "wb") as f:
        pickle.dump(route_info, f)

class CIRPState(object):
    def __init__(self,
                 input: dict,
                 device: str,
                 fname: str = None):
        #-----------
        # locations
        #-----------
        # static 
        self.loc_coords       = input["loc_coords"]         # [batch_size x num_locs x coord_dim]
        self.loc_cap          = input["loc_cap"]            # [batch_size x num_locs]
        self.loc_consump_rate = input["loc_consump_rate"]   # [batch_size x num_locs]
        # dynamic
        self.loc_curr_battery = input["loc_initial_battery"].clone() # [batch_size x num_locs]

        #--------
        # depots
        #--------
        # static
        self.depot_coords = input["depot_coords"] # [batch_size x num_depots x coord_dim]
        self.depot_discharge_rate = input["depot_discharge_rate"] # [batch_size x num_depots]
        # depots whose discharge rate is less than threshold
        self.th = 10.0
        self.small_depots = self.depot_discharge_rate < self.th # [batch_size x num_depots]
        
        #-------------------
        # locations & depot
        #-------------------
        self.coords = torch.cat((self.loc_coords, self.depot_coords), 1)

        #----------
        # vehicles
        #----------
        # TODO: speed should depend on edges, not vehicles
        # static
        self.vehicle_cap = input["vehicle_cap"].clone().detach()   # [batch_size x num_vehicles]
        self.vehicle_discharge_lim = input["vehicle_discharge_lim"].clone().detach() # [batch_size x num_vehicles]
        self.vehicle_discharge_rate = input["vehicle_discharge_rate"] # [batch_size x num_vehicles]
        # dynamic
        self.vehicle_position_id  = input["vehicle_initial_position_id"].clone() # node ids in which vehicles are [batch_size x num_vehicles]
        self.vehicle_curr_battery = input["vehicle_cap"].clone() # initialized with battery fully chareged [batch_size x num_vehicles]
        self.vehicle_unavail_time = torch.zeros(self.vehicle_cap.size(), dtype=torch.float, device=device) # [batch_size x num_vehicles]
        self.wait_vehicle = torch.full(self.vehicle_cap.size(), False, device=device) # [batch_size x num_vehicles] stores whether the vehicle is waiting or not
        self.phase_id = {"move": 0, "pre": 1, "charge": 2, "post": 3}
        self.phase_id_max = max(self.phase_id.values())
        self.vehicle_phase = torch.full(self.vehicle_cap.size(), self.phase_id["post"], dtype=torch.long, device=device) # [batch_size x num_vehicles] # 0 -> "move", 1 -> "charge"
        self.vehicle_consump_rate = input["vehicle_consump_rate"].clone()
        self.vehicle_position_id_prev  = input["vehicle_initial_position_id"].clone() # for visualization
        self.vehicle_move_time = torch.zeros(self.vehicle_cap.size(), dtype=torch.float, device=device)
        self.vehicle_pre_time  = torch.zeros(self.vehicle_cap.size(), dtype=torch.float, device=device)
        self.vehicle_work_time = torch.zeros(self.vehicle_cap.size(), dtype=torch.float, device=device)
        self.vehicle_post_time = torch.zeros(self.vehicle_cap.size(), dtype=torch.float, device=device)
        #-----------
        # paramters
        #-----------
        self.batch_size   = self.loc_coords.size(0)
        self.coord_dim    = self.loc_coords.size(-1)
        self.num_locs     = self.loc_coords.size(1)
        self.num_depots   = self.depot_coords.size(1)
        self.num_vehicles = self.vehicle_cap.size(1)
        self.num_nodes    = self.num_locs + self.num_depots
        self.wait_time    = torch.FloatTensor([input["wait_time"]]).to(device)
        self.time_horizon = torch.FloatTensor([input["time_horizon"]]).to(device)
        self.speed        = (torch.FloatTensor([input["vehicle_speed"]]).to(device) / input["grid_scale"]).squeeze(-1) # [batch_size x 1] -> [batch_size]
        self.max_cap      = torch.max(torch.tensor([torch.max(self.loc_cap).item(), torch.max(self.vehicle_cap).item()])).to(device) 
        self.device       = device
        self.loc_min_battery = 0.0 # TODO
        depot_prepare_time = 0.17 # 10 mins
        loc_prepare_time   = 0.5  # 30 mins
        self.pre_time_depot  = torch.FloatTensor([depot_prepare_time]).to(device)
        self.post_time_depot = torch.FloatTensor([depot_prepare_time]).to(device)
        self.post_time_loc   = torch.FloatTensor([loc_prepare_time]).to(device)
        self.pre_time_loc    = torch.FloatTensor([loc_prepare_time]).to(device)
        self.return_depot_within_time_horizon = False
        
        #-------
        # utils
        #-------
        self.loc_arange_idx     = torch.arange(self.num_locs).to(self.device).unsqueeze(0).expand(self.batch_size, -1)
        self.depot_arange_idx   = torch.arange(self.num_locs, self.num_nodes).to(self.device).unsqueeze(0).expand(self.batch_size, -1)
        self.node_arange_idx    = torch.arange(self.num_nodes).to(self.device).unsqueeze(0).expand(self.batch_size, -1)
        self.vehicle_arange_idx = torch.arange(self.num_vehicles).to(self.device).unsqueeze(0).expand(self.batch_size, -1)
        
        #----------------------
        # common dynamic state
        #----------------------
        self.next_vehicle_id = torch.zeros(self.batch_size, dtype=torch.long, device=device) # firstly allocate 0-th vehicles
        self.skip = torch.full((self.batch_size,), False, dtype=bool, device=device) # [batch_size]
        self.end  = torch.full((self.batch_size,), False, dtype=bool, device=device) # [batch_size]
        self.current_time = torch.zeros(self.batch_size, dtype=torch.float, device=device) # [bath_size]
        self.tour_length = torch.zeros(self.batch_size, dtype=torch.float, device=device) # [batch_size]
        self.penalty_empty_locs = torch.zeros(self.batch_size, dtype=torch.float, device=device) # [batch_size]
        next_vehicle_mask = torch.arange(self.num_vehicles).to(self.device).unsqueeze(0).expand(self.batch_size, -1).eq(self.next_vehicle_id.unsqueeze(-1)) # [batch_size x num_vehicles]
        self.mask = self.update_mask(self.vehicle_position_id[next_vehicle_mask], next_vehicle_mask)
        self.charge_queue = torch.zeros((self.batch_size, self.num_depots, self.num_vehicles), dtype=torch.long, device=device)

        #-------------------
        # for visualization
        #-------------------
        self.fname = fname
        self.episode_step = 0
        if fname is not None:
            self.vehicle_batt_history = [[[] for __ in range(self.num_vehicles)] for _ in range(self.batch_size)]
            self.loc_batt_history = [[[] for __ in range(self.num_locs)] for _ in range(self.batch_size)]
            self.time_history = [[] for _ in range(self.batch_size)]
            self.down_history = [[] for _ in range(self.batch_size)]
            # visualize initial state
            all_batch = torch.full((self.batch_size, ), True, device=self.device)
            self.visualize_state_batch(all_batch)

    def reset(self, input):
        self.__init__(input)

    def get_coordinates(self, node_id: torch.Tensor):
        """
        Paramters
        ---------
        node_id: torch.LongTensor [batch_size]

        Returns
        -------
        coords: torch.FloatTensor [batch_size x coord_dim]
        """
        return self.coords.gather(1, node_id[:, None, None].expand(self.batch_size, 1, self.coord_dim)).squeeze(1)
    
    def is_depot(self, node_id: torch.Tensor):
        """
        Paramters
        ---------
        node_id: torch.Tensor [batch_size]:
        """
        return node_id.ge(self.num_locs)

    def get_loc_mask(self, node_id: torch.Tensor):
        """
        Paramters
        ---------
        node_id: torch.Tensor [batch_size]:
        """
        return self.loc_arange_idx.eq(node_id.unsqueeze(-1))

    def get_depot_mask(self, node_id: torch.Tensor):
        """
        Paramters
        ---------
        node_id: torch.Tensor [batch_size]:
        """
        return self.depot_arange_idx.eq(node_id.unsqueeze(-1))
    
    def get_vehicle_mask(self, vehicle_id: torch.Tensor):
        """
        Paramters
        ---------
        vehicle_id: torch.Tensor [batch_size]:
        """
        return self.vehicle_arange_idx.eq(vehicle_id.unsqueeze(-1))

    def get_curr_nodes(self):
        return self.vehicle_position_id[self.get_vehicle_mask(self.next_vehicle_id)]

    def update(self,
               next_node_id: torch.Tensor):
        """
        Paramters
        ---------
        next_node_id: torch.LongTensor [batch_size]
            ids of nodes where the currently selected vehicles visit next
        """
        curr_vehicle_id = self.next_vehicle_id # [batch_size]
        curr_vehicle_mask = self.get_vehicle_mask(curr_vehicle_id) # [batch_size x num_vehicles]
        # assert (self.vehicle_phase[curr_vehicle_mask] == self.phase_id["post"]).sum() == self.batch_size, "all sected vehicles should be in post phase"

        #-------------------------------------------
        # update currrently selected vehicle's plan
        #-------------------------------------------
        # calculate travel distance & time of the currently selected vehicle
        curr_node_id = self.vehicle_position_id.gather(-1, curr_vehicle_id.unsqueeze(-1)).squeeze(-1) # [batch_size]
        curr_coords = self.get_coordinates(curr_node_id) # [batch_size x coord_dim]
        next_coords = self.get_coordinates(next_node_id) # [batch_size x coord_dim]
        travel_distance = torch.linalg.norm(curr_coords - next_coords, dim=-1) # [batch_size]
        travel_time = travel_distance / self.speed # [batch_size]

        # check waiting vehicles
        do_wait = (curr_node_id == next_node_id) & ~self.skip # [batch_size] wait: stay at the same place
        self.wait_vehicle.scatter_(-1, index=curr_vehicle_id.unsqueeze(-1), src=do_wait.unsqueeze(-1))

        # update the plan of the selected vehicles
        self.vehicle_unavail_time.scatter_(-1, curr_vehicle_id.unsqueeze(-1), travel_time.unsqueeze(-1))
        self.vehicle_position_id_prev.scatter_(-1, curr_vehicle_id.unsqueeze(-1), curr_node_id.unsqueeze(-1))
        self.vehicle_position_id.scatter_(-1, curr_vehicle_id.unsqueeze(-1), next_node_id.unsqueeze(-1))
        self.vehicle_phase.scatter_(-1, curr_vehicle_id.unsqueeze(-1), self.phase_id["move"])

        #---------------------------
        # estimate store phase time
        #---------------------------
        # moving 
        self.vehicle_move_time.scatter_(-1, curr_vehicle_id.unsqueeze(-1), travel_time.unsqueeze(-1)) # [batch_size x num_vehicles]
        
        # pre/post-operation time
        at_depot = self.is_depot(next_node_id).unsqueeze(-1)
        at_loc = ~at_depot
        curr_vehicle_at_loc = curr_vehicle_mask & at_loc
        curr_vehicle_at_depot = curr_vehicle_mask & at_depot
        self.vehicle_pre_time  += curr_vehicle_at_loc * self.pre_time_loc + curr_vehicle_at_depot * self.pre_time_depot
        self.vehicle_post_time += curr_vehicle_at_loc * self.post_time_loc + curr_vehicle_at_depot * self.post_time_depot
        
        # charge/supply time
        destination_loc_mask = self.get_loc_mask(next_node_id) # [batch_size x num_locs]
        at_loc_sqz = at_loc.squeeze(-1) # [batch_size]
        #-------------------------------------
        # supplying time (visiting locations)
        #-------------------------------------
        unavail_depots = self.get_unavail_depots2(next_node_id).unsqueeze(-1).expand_as(self.depot_coords)
        depot_coords = self.depot_coords + 1e+6 * unavail_depots
        loc2depot_min = torch.linalg.norm(self.get_coordinates(next_node_id).unsqueeze(1) - depot_coords, dim=-1).min(-1)[0] # [batch_size]
        discharge_lim = torch.maximum(loc2depot_min.unsqueeze(-1) * self.vehicle_consump_rate, self.vehicle_discharge_lim) # [batch_size x num_vehicles]
        veh_discharge_lim = (self.vehicle_curr_battery - (travel_distance.unsqueeze(-1) * self.vehicle_consump_rate) - discharge_lim).clamp(0.0)
        demand_on_arrival = torch.minimum(((self.loc_cap - (self.loc_curr_battery - self.loc_consump_rate * travel_time.unsqueeze(-1))) * destination_loc_mask).sum(-1, keepdim=True), 
                                          veh_discharge_lim) # [batch_size x num_vehicles]
        # split supplying TODO: need clippling ?
        charge_time_tmp = demand_on_arrival / (self.vehicle_discharge_rate - (self.loc_consump_rate * destination_loc_mask).sum(-1, keepdim=True)) # [batch_sizee x num_vehicles]
        cannot_supplly_full = ((veh_discharge_lim - charge_time_tmp * self.vehicle_discharge_rate) < 0.0) # [batch_size x num_vehicles]
        next_vehicles_sd  = curr_vehicle_at_loc & cannot_supplly_full  # vehicles that do split-delivery [batch_size x num_vehicles]
        next_vehicles_nsd = curr_vehicle_at_loc & ~cannot_supplly_full # vehicles that do not split-delivery [batch_size x num_vehicles]
        charge_time = (charge_time_tmp * next_vehicles_nsd).sum(-1) # [batch_size]
        charge_time += ((veh_discharge_lim / self.vehicle_discharge_rate) * next_vehicles_sd).sum(-1) # [batch_size]
        #---------------------------------
        # charging time (visiting depots)
        #---------------------------------
        curr_depot_mask = self.get_depot_mask(next_node_id) # [batch_size x num_depots]
        charge_time += (((self.vehicle_cap - self.vehicle_curr_battery) / ((self.depot_discharge_rate * curr_depot_mask).sum(-1, keepdim=True) + SMALL_VALUE)) * curr_vehicle_at_depot).sum(-1) # charge time for split supplying (loc will not be fully [charged)
        #--------------------------------------------------------
        # update unavail_time (charge_time) of selected vehicles
        #--------------------------------------------------------
        self.vehicle_work_time += charge_time.clamp(0.0).unsqueeze(-1) * (curr_vehicle_mask & ~self.wait_vehicle) # [charging_batch_size]
        self.vehicle_work_time += self.wait_time * (curr_vehicle_mask & self.wait_vehicle) # waiting vehicles
        
        #----------------------------------------------------------------------
        # select a vehicle that we determine its plan while updating the state
        # (greddy approach: select a vehicle whose unavail_time is minimum)
        #----------------------------------------------------------------------
        # align the phase of the selected vehicles to "post"
        num_not_post = 1 # temporaly initial value
        while num_not_post > 0:
            vechicle_unavail_time_min, next_vehicle_id = self.vehicle_unavail_time.min(dim=-1) # [batch_size], [batch_size]
            next_vehicle_mask = self.get_vehicle_mask(next_vehicle_id) # [batch_size x num_vehicles]
            not_post_batch = (self.vehicle_phase[next_vehicle_mask] != self.phase_id["post"]) # [batch_size]
            num_not_post = (not_post_batch).sum()
            if num_not_post > 0:
                self.update_state(vechicle_unavail_time_min, self.vehicle_position_id[next_vehicle_mask], next_vehicle_id, next_vehicle_mask, not_post_batch, align_phase=True)
        # now, all the vehicle selected in all the batchs should be in charge phase
        # update the state at the time when the selected vehicles finish charging
        vechicle_unavail_time_min, next_vehicle_id = self.vehicle_unavail_time.min(dim=-1) # [batch_size], [batch_size]
        self.next_vehicle_id = next_vehicle_id
        next_vehicle_mask = self.get_vehicle_mask(next_vehicle_id) # [batch_size x num_vehicles]
        next_node_id = self.vehicle_position_id[next_vehicle_mask]
        all_batch = torch.full((self.batch_size, ), True, device=self.device)
        self.update_state(vechicle_unavail_time_min, next_node_id, next_vehicle_id, next_vehicle_mask, all_batch, align_phase=False)

        #-------------
        # update mask
        #-------------
        self.mask = self.update_mask(next_node_id, next_vehicle_mask)

        #--------------------------
        # validation check of mask
        #--------------------------
        all_zero = self.mask.sum(-1) == 0 # [batch_size]
        assert not all_zero.any(), "there is no node that the vehicle can visit!"

    def update_state(self, 
                     elapsed_time: torch.Tensor,
                     next_node_id: torch.Tensor,
                     next_vehicle_id: torch.Tensor, 
                     next_vehicle_mask: torch.Tensor, 
                     update_batch: torch.Tensor,
                     align_phase: bool):
        """
        Parameters
        ----------
        elapsed_time: torch.FloatTensor [batch_size]
        next_node_id: torch.LongTensor [batch_size]
        next_vehicle_id: torch.LongTensor [batch_size]
        next_vehicle_mask: torch.BoolTensor [batch_size x num_vehicles]
        update_batch: torch.BoolTensor [batch_size]
        align_phase: bool
        """
        #-------------------
        # clip elapsed_time
        #-------------------
        remaing_time = (self.time_horizon - self.current_time).clamp(0.0) # [batch_size]
        elapsed_time = torch.minimum(remaing_time, elapsed_time)
        
        #---------------------------------------------
        # moving vehicles (consuming vehicle battery)
        #---------------------------------------------
        moving_vehicles = (self.vehicle_phase == self.phase_id["move"]) & update_batch.unsqueeze(-1) # [batch_size x num_vehicles]
        moving_not_wait_vehicles = moving_vehicles & ~self.wait_vehicle # [batch_size x num_vehicles]
        self.vehicle_curr_battery -= self.vehicle_consump_rate * (self.speed * elapsed_time).unsqueeze(-1) * moving_not_wait_vehicles # Travel battery consumption
        # update total tour length
        # self.tour_length[update_batch] += moving_vehicles.sum(-1)[update_batch] * self.speed[update_batch] * elapsed_time[update_batch]
        self.tour_length += moving_vehicles.sum(-1) * self.speed * elapsed_time * update_batch.float()

        #-------------------------------
        # charging / supplying vehicles 
        #-------------------------------
        at_depot = self.is_depot(self.vehicle_position_id) # [batch_size x num_vehicles]
        charge_phase_vehicles = (self.vehicle_phase == self.phase_id["charge"]) # [batch_size x num_vehicles]
        #-----------------------------
        # charging (depot -> vehicle)
        #-----------------------------
        queued_vehicles = self.charge_queue.sum(1) > 1 # [batch_size x num_vehicles]
        charging_vehicles = charge_phase_vehicles & at_depot & update_batch.unsqueeze(-1) & ~queued_vehicles # [batch_size x num_vehicles]
        charging_vehicle_position_idx = (self.vehicle_position_id - self.num_locs) * charging_vehicles.long()
        self.vehicle_curr_battery += self.depot_discharge_rate.gather(-1, charging_vehicle_position_idx) * elapsed_time.unsqueeze(-1) * charging_vehicles.float() # [batch_size x num_vehicles]
        #---------------------------------
        # supplying (vehicle -> location)
        #---------------------------------
        supplying_vehicles = charge_phase_vehicles & ~at_depot & update_batch.unsqueeze(-1) # [batch_size x num_vehicles]
        # location battery charge
        # NOTE: In locs where a vehicle is staying, the battery of the locs incrases by loc_consump_rate * elapsed_time, not vehicle_discarge_rate * elasped_time.
        # However, as the battery of the locs should be full when a vehicle is staying and it is clamped by max_cap later, we ignore this mismatch here.
        supplying_vehicle_position_idx = self.vehicle_position_id * supplying_vehicles.long() # [batch_size x num_vehicles]
        self.loc_curr_battery.scatter_reduce_(-1, 
                                                supplying_vehicle_position_idx, 
                                                self.vehicle_discharge_rate * elapsed_time.unsqueeze(-1) * supplying_vehicles.float(), 
                                                reduce="sum")
        # vechicle battery consumption (consumption rate is different b/w waiting vehicles and not waiting ones)
        # not waiting
        supplying_not_wait_vehicles = supplying_vehicles & ~self.wait_vehicle # [batch_size x num_vehicles]
        self.vehicle_curr_battery -= self.vehicle_discharge_rate * elapsed_time.unsqueeze(-1) * supplying_not_wait_vehicles.float()
        # waiting
        supplying_wait_vehicles = supplying_vehicles & self.wait_vehicle
        supplying_vehicle_position_idx_ = self.vehicle_position_id * supplying_wait_vehicles.long()
        self.vehicle_curr_battery -= self.loc_consump_rate.gather(-1, supplying_vehicle_position_idx_) * elapsed_time.unsqueeze(-1) * (supplying_wait_vehicles).float() # [batch_size x num_vehicles]
        
        #----------------------------------
        # battery consumption of locations
        #----------------------------------
        self.loc_curr_battery -= self.loc_consump_rate * (elapsed_time * update_batch.float()).unsqueeze(-1)
        
        # TODO:
        # print(self.vehicle_curr_battery[self.vehicle_curr_battery<0])
        # location battery is always greater (less) than 0 (capacity)
        self.vehicle_curr_battery = self.vehicle_curr_battery.clamp(min=0.0)
        self.vehicle_curr_battery = self.vehicle_curr_battery.clamp(max=self.vehicle_cap)

        #----------------
        # update penalty
        #----------------
        down_locs = (self.loc_curr_battery - self.loc_min_battery) <= 0.0 # SMALL_VALUE [batch_size x num_locs]
        # ignore penalty in skipped episodes
        num_empty_locs = ((-self.loc_curr_battery + self.loc_min_battery) / self.loc_consump_rate) * down_locs * (~self.skip.unsqueeze(-1)) # [batch_size x num_locs]
        # empty_locs = ((-self.loc_curr_battery + self.loc_min_battery)[down_locs] / self.loc_consump_rate[down_locs]) # 1d
        # num_empty_locs = torch.zeros((self.batch_size, self.num_locs), dtype=torch.float, device=self.device).masked_scatter_(down_locs, empty_locs)
        # num_empty_locs[self.skip] = 0.0 # ignore penalty in skipped episodes
        self.penalty_empty_locs += num_empty_locs.sum(-1) * update_batch / self.num_locs # [batch_size]
        # location battery is always greater (less) than minimum battery (capacity)
        self.loc_curr_battery = self.loc_curr_battery.clamp(min=self.loc_min_battery)
        self.loc_curr_battery = self.loc_curr_battery.clamp(max=self.loc_cap)

        #---------------------
        # update unavail_time
        #---------------------
        # decrease unavail_time
        queued_vehicles = self.charge_queue.sum(1) > 1 # [batch_size x num_vehicles]
        update_vehicles = ~queued_vehicles & update_batch.unsqueeze(-1) # [batch_size x num_vehicles]
        self.vehicle_unavail_time -= elapsed_time.unsqueeze(-1) * update_vehicles
        
        #---------------------
        # update current time
        #---------------------
        self.current_time += elapsed_time * update_batch

        #---------------------
        # visualize the state
        #---------------------
        if self.fname is not None:
            vis_batch = update_batch & ~self.skip & (torch.abs(elapsed_time) > SMALL_VALUE)
            self.visualize_state_batch(vis_batch)

        # update unavail time
        if align_phase:
            at_depot = self.is_depot(next_node_id).unsqueeze(-1) # [batch_size x 1]
            next_vehicles_on_update_batch = next_vehicle_mask & update_batch.unsqueeze(-1) # [batch_size x num_vehicles]
            # clear (zero out) unavail time of next vehicles on updated batch
            self.vehicle_unavail_time *= ~next_vehicles_on_update_batch # 0 or 1 [batch_size x num_vehicles]
            
            #-------------------------
            # moving -> pre operation
            #-------------------------
            next_vehicle_on_move = next_vehicles_on_update_batch & (self.vehicle_phase == self.phase_id["move"]) # [batch_size x num_vehicles]
            if next_vehicle_on_move.sum() > 0:
                self.vehicle_unavail_time += self.vehicle_pre_time * next_vehicle_on_move # [batch_size x num_vehicles]
                self.vehicle_move_time *= ~next_vehicle_on_move # reset move time
                # add supplying vehicles to charge-query
                head = self.charge_queue.min(-1)[0] # [batch_size x num_depots]
                destination_depot_mask = self.get_depot_mask(next_node_id) # [batch_size x num_depots]
                update_query_mask = (~self.wait_vehicle & next_vehicle_on_move).unsqueeze(1) & destination_depot_mask.unsqueeze(-1) # [batch_size x num_depots x num_vehicles]
                # self.charge_queue[update_query_mask] = head.unsqueeze(-1).expand_as(self.charge_queue)[update_query_mask] + 1
                self.charge_queue += (head + 1).unsqueeze(-1) * update_query_mask # [batch_size x num_depots x num_vehicles]
            
            #---------------------------
            # pre operation -> charging
            #---------------------------
            next_vehicle_on_pre = next_vehicles_on_update_batch & (self.vehicle_phase == self.phase_id["pre"]) # [batch_size x num_vehicles]
            if next_vehicle_on_pre.sum() > 0:
                self.vehicle_unavail_time += self.vehicle_work_time * next_vehicle_on_pre
                self.vehicle_pre_time *= ~next_vehicle_on_pre # reset pre time
            
            #----------------------------
            # charging -> post operation
            #----------------------------
            next_vehicle_on_charge = next_vehicles_on_update_batch & (self.vehicle_phase == self.phase_id["charge"]) # [batch_size x num_vehicles]
            if next_vehicle_on_charge.sum() > 0:
                self.vehicle_unavail_time += self.vehicle_post_time * next_vehicle_on_charge
                self.vehicle_work_time *= ~next_vehicle_on_charge

            #--------------
            # update phase
            #--------------
            self.vehicle_phase += next_vehicles_on_update_batch.long()
        else:
            #----------------------------------------------
            # post operation -> move (determine next node)
            #----------------------------------------------
            # update charge-queue
            # do not change charge queue when the next vehicle is waiting
            # because the waiting vehicles are not added to the queue
            destination_depot_mask = self.get_depot_mask(next_node_id) & ~self.wait_vehicle[next_vehicle_mask].unsqueeze(-1) # [batch_size x num_depots]
            # self.charge_queue[destination_depot_mask] -= 1 # [batch_size x num_depots x num_vehicles]
            self.charge_queue -= destination_depot_mask.long().unsqueeze(-1) # [batch_size x num_depots x num_vehicles]
            self.charge_queue = self.charge_queue.clamp(0)
            # reset the waiting flag of the selected vehicles
            self.wait_vehicle.scatter(-1, next_vehicle_id.to(torch.int64).unsqueeze(-1), False)
            # vehiclel_unavail_time of the selected vehicle is updated with travel time in the early step of next called self.update
            self.vehicle_post_time *= ~next_vehicle_mask
        
        #-------------------
        # update skip batch
        #-------------------
        # end flags
        self.end = self.current_time >= self.time_horizon # [batch_size]
        # skip
        self.skip = self.end # [batch_size]
        
    #---------
    # masking
    #---------
    def update_mask(self, next_node_id, next_vehicle_mask):
        #---------------------------------------------------------------
        # mask: 0 -> infeasible, 1 -> feasible [batch_size x num_nodes]
        #---------------------------------------------------------------
        mask = torch.ones(self.batch_size, self.num_nodes, dtype=torch.int32, device=self.device) # [batch_size x num_nodes]
        # mask 0: if a selected vehicle is out of battery, we make it return to a depot
        self.mask_unreturnable_nodes(mask, next_node_id, next_vehicle_mask)
        # mask 2: forbits vehicles to move between two different depots
        # i.e., if a selcted vechile is currently at a depot, it cannot visit other depots in the next step (but it can stay in the same depot)
        # self.mask_depot_to_other_depots(mask, next_node_id)
        # mask 3: vehicles cannot visit a location/depot that other vehicles are visiting
        self.mask_visited_locs(mask)
        # mask 4: forbit vehicles to visit depots that have small discharge rate
        self.remove_small_depots(mask, next_node_id)
        # mask 5: in skipped episodes(instances), the selcted vehicles always stay in the same place
        self.mask_skipped_episodes(mask, next_node_id)
        return mask

    def mask_unreturnable_nodes(self, mask, next_node_id, next_vehicle_mask):
        """
        There are two patterns:
            1. unreturnable to depot within time horizon
            2. unreturnable to depot without running out of vehicle battery
        """
        # mask 1: guarantee that all the vehicles return to depots within the time horizon
        remaining_time = (self.time_horizon - self.current_time).unsqueeze(-1).clamp(0.0) # [batch_size x 1]
        current_coord = torch.gather(self.coords, 1, next_node_id.view(-1, 1, 1).expand(-1, 1, self.coord_dim)) # [batch_size, 1, coord_dim]
        unavail_depots = self.small_depots.unsqueeze(-1).expand_as(self.depot_coords) # [batch_size x num_depots x coord_dim]
        depot_coords = self.depot_coords + 1e+6 * unavail_depots # set large value for removing small depots [batch_size x num_depots x coord_dim]
        current_to_loc = torch.linalg.norm(self.loc_coords - current_coord, dim=-1) # [batch_size x num_locs]
        loc_to_depot = torch.min(torch.cdist(self.loc_coords, depot_coords), -1)[0] # travel time b/w locs and the nearest depot [batch_size x num_locs]
        current_to_loc_time = current_to_loc / self.speed.unsqueeze(-1)
        loc_to_depot_time = loc_to_depot / self.speed.unsqueeze(-1)
        wait_time = self.wait_time * (torch.abs(current_to_loc) < SMALL_VALUE) # [batch_size x num_locs]

        #--------------------------------------------------------------------------------------------------------------
        # vehicles can visit only locations that the vehicles can return to depots within time horizon after the visit
        # i.e. travel time t_(current_node -> next_loc -> depot) + current_time <= time_horizon
        #--------------------------------------------------------------------------------------------------------------
        # loc_charge_time should be wait_time not zero in the locations where a vehicle is waiting,
        # but ignore that because those locations are masked out later 
        runout_battery_loc = ((current_to_loc + loc_to_depot) * self.vehicle_consump_rate[next_vehicle_mask].unsqueeze(-1) # travel consumption
                            #    + self.vehicle_discharge_rate[next_vehicle_mask].unsqueeze(-1) * loc_charge_time # supply consumption: curr_demand + loc_charge_time * self.loc_consump_rate
                               + self.loc_consump_rate * wait_time # supply consumption when waiting
                             ) > self.vehicle_curr_battery[next_vehicle_mask].unsqueeze(-1) + BIT_SMALL_VALUE
        
        # if its battery is zero, the vehicle should return to a depot (this is used only when vehicle_consump_rate = 0)
        battery_zero = torch.abs(self.vehicle_curr_battery[next_vehicle_mask]) < SMALL_VALUE # [batch_size]
        # mask for unreturnable locations
        unreturnable_loc = battery_zero.unsqueeze(-1) | runout_battery_loc # [batch_size x num_locs]
        if self.return_depot_within_time_horizon:
            # ignore the battery change of visited locations (either way, they are masked out later)
            loc_battery_on_arrival = (self.loc_curr_battery - self.loc_consump_rate * (current_to_loc_time + self.pre_time_loc)).clamp(self.loc_min_battery) # [batch_size x num_locs]
            curr_demand = torch.minimum(self.loc_cap - loc_battery_on_arrival, self.vehicle_curr_battery[next_vehicle_mask].unsqueeze(-1)) # [batch_size x num_locs]
            # time limit
            loc_charge_time = curr_demand / (self.vehicle_discharge_rate[next_vehicle_mask].unsqueeze(-1) - self.loc_consump_rate) # [batch_size x num_locs]
            exceed_timehorizon_loc = (current_to_loc_time + self.pre_time_loc + loc_charge_time + self.post_time_loc + loc_to_depot_time + wait_time).gt(remaining_time + BIT_SMALL_VALUE)
            unreturnable_loc |= exceed_timehorizon_loc
        
        #---------------------------------------------------------------------------------------
        # vehicles can visit only depots that the vehicles can arrive there within time horizon
        # i.e. travel time t_(current_node -> depot) + current_time <= time_horizon
        #---------------------------------------------------------------------------------------
        unavail_depots2 = self.get_unavail_depots(next_node_id).unsqueeze(-1).expand_as(self.depot_coords) # [batch_size x num_depots x coord_dim]
        depot_coords2 = self.depot_coords + 1e+6 * unavail_depots2
        current_to_depot = torch.linalg.norm(depot_coords2 - current_coord, dim=-1) # [batch_size x num_depots]
        current_to_depot_time = current_to_depot / self.speed.unsqueeze(-1) # [batch_size x n_depots]
        # battery
        curr2depot_batt = current_to_depot * self.vehicle_consump_rate[next_vehicle_mask].unsqueeze(-1) # [batch_size x num_depots]
        veh_batt = self.vehicle_curr_battery[next_vehicle_mask].unsqueeze(-1) # [batch_size x 1]
        runout_battery_depot = curr2depot_batt >= veh_batt + BIT_SMALL_VALUE # [batch_size x num_depots]
        unreturnable_depot = runout_battery_depot
        if self.return_depot_within_time_horizon:
            # time
            exceed_timehorizon_depot = (current_to_depot).gt(remaining_time + BIT_SMALL_VALUE) # [batch_size x num_depots]
            unreturnable_depot |= exceed_timehorizon_depot

        # there should be at least one depot that the vehicle can reach
        i = 0; atol_list=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] # TODO: float -> double
        while (((~unreturnable_depot).int().sum(-1) == 0) & ~self.skip).any():
            all_zero = ((~unreturnable_depot).int().sum(-1) == 0).unsqueeze(-1) # [batch_size x 1]
            close_to_runout_battery = torch.isclose(curr2depot_batt, veh_batt, atol=atol_list[i])
            if self.return_depot_within_time_horizon:
                close_to_timehorizon = torch.isclose(current_to_depot_time, remaining_time, atol=atol_list[i]) # [batch_size x num_depots]
                close_to_runout_battery |= close_to_timehorizon
            unreturnable_depot[all_zero & close_to_runout_battery] = False
            i += 1
            if i > len(atol_list):
                print(self.depot_discharge_rate[all_zero.squeeze(-1)])
                print(f"battery consumption b/w loc2depot: {(current_to_depot * self.vehicle_consump_rate[next_vehicle_mask].unsqueeze(-1))[all_zero.squeeze(-1)].tolist()}")
                print(f"selected vehicle's battery: {self.vehicle_curr_battery[next_vehicle_mask].unsqueeze(-1)[all_zero].tolist()}")
                print(torch.where(all_zero))
                print(f"travel time b/w loc2depot: {current_to_depot_time[all_zero.squeeze(-1)].tolist()}")
                print(f"remaining time: {remaining_time[all_zero].tolist()}")
                assert False, "some vehicles could not return to any depots within time horizon due to numerical error."
        
        #--------------------------
        # update unreturnable mask
        #--------------------------
        unreturnable_mask = torch.cat((unreturnable_loc, unreturnable_depot), 1) # [batch_size x num_nodes]
        mask *= ~unreturnable_mask
    
    def mask_visited_locs(self, mask: torch.Tensor):
        """
        Remove visited locs from the next-node candidates
        
        Parameters
        ----------
        mask: torch.LongTensor [batch_size x num_nodes]
        """
        reserved_loc = torch.full((self.batch_size, self.num_nodes), False, device=self.device)
        reserved_loc.scatter_(1, self.vehicle_position_id, True)
        reserved_loc[:, self.num_locs:] = False
        mask *= ~reserved_loc

    def mask_depot_to_other_depots(self, mask: torch.Tensor, next_node_id: torch.Tensor):
        """
        A mask for removing moving between different two depots
        
        Parameters
        ----------
        mask: torch.LongTesnor [batch_size x num_nodes]
        next_node_id: torch.LongTensor [batch_size]
        """
        at_depot = next_node_id.ge(self.num_locs).unsqueeze(1) # [batch_size x 1]
        other_depot = self.node_arange_idx.ne(next_node_id.unsqueeze(1)) # [batch_size x num_nodes]
        other_depot[:, :self.num_locs] = False # ignore locations here
        mask *= ~(at_depot & other_depot)

    def remove_small_depots(self, mask: torch.Tensor, next_node_id: torch.Tensor):
        """
        A mask for removing small depots, which have low discharge_rate
        """
        unavail_depots = self.get_unavail_depots(next_node_id) # [batch_size x num_depots]
        unavail_nodes = torch.cat((torch.full((self.batch_size, self.num_locs), False).to(self.device), unavail_depots), -1) # [batch_size x num_nodes]
        mask *= ~unavail_nodes
        return mask
    
    def get_unavail_depots(self, next_node_id: torch.Tensor):
        stayed_depots = self.get_depot_mask(next_node_id) # [batch_size x num_depots]
        # if the initial depot is a small depot, the vehicle can stay there
        return ~stayed_depots & self.small_depots # (small depots) and (not visited) [batch_size x num_depots]

    def get_unavail_depots2(self, next_node_id: torch.Tensor):
        return self.small_depots
    
    def mask_skipped_episodes(self, mask: torch.Tensor, next_node_id: torch.Tensor):
        current_node = self.node_arange_idx.eq(next_node_id.unsqueeze(1)).int() # [batch_size x num_nodes]
        mask[self.skip] = current_node[self.skip]

    def get_inputs(self):
        """
        Returns
        -------
        node_feats: torch.tensor [batch_size x num_nodes x node_feat_dim]
            input features of nodes
        vehicle_feats: torch.tensor [batch_size x num_vehicles x vechicle_feat_dim]
            input features of vehicles
        """
        visit_mask = torch.full((self.batch_size, self.num_nodes), 0.0, device=self.device)
        visit_mask.scatter_(1, self.vehicle_position_id, 1.0)
        # for locations (loc_dim = 1+2+1+1+1+1 = 7)
        loc_feats = torch.cat((
            visit_mask[:, :self.num_locs, None], # visited by an EV?
            self.loc_coords,  # [batch_size x num_locs x coord_dim]
            self.loc_cap.unsqueeze(-1) / self.max_cap, # [batch_size x num_locs x 1]
            self.loc_consump_rate.unsqueeze(-1) / self.max_cap, # [batch_size x num_locs x 1]
            self.loc_curr_battery.unsqueeze(-1) / self.max_cap,  # [batch_size x num_locs x 1]
            (self.loc_curr_battery / self.loc_consump_rate).unsqueeze(-1) # expected time to go down [batch_size x num_locs x 1]
        ), -1)
        # for depots (depot_dim = 1+2+1 = 4)
        depot_feats = torch.cat((
            visit_mask[:, self.num_locs:, None], # visited by an EV?
            self.depot_coords, # [batch_size x num_depots x coord_dim]
            self.depot_discharge_rate.unsqueeze(-1) / self.max_cap # [batch_size x num_depots x 1]
        ), -1)
        # for vehicles (vehicle_dim = 1+2+1+1++4+1+1 = 11)
        curr_vehicle_coords = self.coords.gather(1, self.vehicle_position_id.unsqueeze(-1).expand(self.batch_size, self.num_vehicles, self.coord_dim)) # [batch_size x num_vehicles x coord_dim]
        vehicle_phase_time = torch.concat((
            self.vehicle_move_time.unsqueeze(-1),
            self.vehicle_pre_time.unsqueeze(-1),
            self.vehicle_work_time.unsqueeze(-1),
            self.vehicle_post_time.unsqueeze(-1)
        ), -1) # [batch_size x num_vehicles x 4]
        vehicle_phase_time.scatter_(-1, self.vehicle_phase.unsqueeze(-1), self.vehicle_unavail_time.unsqueeze(-1)) # [batch_size x num_vehicles x 4]
        vehicle_phase_time_sum = vehicle_phase_time.sum(-1, keepdim=True)
        vehicle_feats = torch.cat((
            self.vehicle_cap.unsqueeze(-1) / self.max_cap, # [batch_size x num_vehicles] -> [batch_size x num_vehicles x 1]
            curr_vehicle_coords, # [batch_size x num_vehicles x coord_dim]
            self.is_depot(self.vehicle_position_id).unsqueeze(-1).to(torch.float), # [batch_size x num_vehicles x 1] at {depot: 0, locations: 1}
            self.vehicle_phase.unsqueeze(-1) / self.phase_id_max, # [batch_size x num_vehicles] -> [batch_size x num_vehicles x 1] the phase of vehicles
            vehicle_phase_time, # remaining time of move, pre-operation, charge/supply, and post-operation [batch_size x num_vehicles x 4]
            vehicle_phase_time_sum, # / (vehicle_phase_time_sum.max(-1, keepdim=True)[0] + SMALL_VALUE), # total unavail time [batch_size x num_vehicles x 1]
            self.vehicle_curr_battery.unsqueeze(-1) / self.max_cap # [batch_size x num_vehicles x 1]
            # self.current_time[:, None, None].expand(-1, self.num_vehicles, 1) / self.time_horizon # [batch_size] -> [batch_size x num_vehicles x 1] 
        ), -1)
        return loc_feats, depot_feats, vehicle_feats

    def get_mask(self):
        """
        Returns
        -------
        mask: torch.tensor [batch_size x num_nodes]
        """
        return self.mask

    def get_selected_vehicle_id(self):
        """
        Returns 
        -------
        next_vehicle_id [batch_size]
        """
        return self.next_vehicle_id.to(torch.int64)

    def all_finished(self):
        """
        Returns
        -------
        end: torch.tensor [batch_size]
        """
        return self.end.all()

    def get_rewards(self):
        # compute the last penalty
        # remaining_time = self.time_horizon - self.current_time # [batch_size]
        # self.loc_curr_battery -= self.loc_consump_rate * remaining_time.unsqueeze(-1)
        # down_locs = (self.loc_curr_battery - self.loc_min_battery) < SMALL_VALUE
        # num_empty_locs = down_locs.count_nonzero(-1) * remaining_time # [batch_size]
        # num_empty_locs = ((self.loc_curr_battery - self.loc_min_battery)[down_locs] / self.loc_consump_rate[down_locs]).sum(-1)
        # num_empty_locs[self.skip] = 0 # ignore penalty in skipped episodes
        # self.penalty_empty_locs += num_empty_locs / self.num_locs # [batch_size]
        # normalization
        penalty = self.penalty_empty_locs / self.time_horizon
        tour_length = self.tour_length / self.num_vehicles
        return {"tour_length": tour_length, "penalty": penalty}

    def visualize_state_batch(self, visualized_batch: torch.BoolTensor):
        if self.episode_step == 0 or UNEQUAL_INTERVAL:
            for batch in range(1):
                if visualized_batch[batch] == False:
                    continue
                self.visualize_state(batch, 
                                    self.current_time[batch].item(), 
                                    self.vehicle_curr_battery[batch], 
                                    self.loc_curr_battery[batch], 
                                    ((self.loc_curr_battery[batch] - self.loc_min_battery) <= 0.0).sum().item(),
                                    self.vehicle_unavail_time[batch])
        else:
            for batch in range(1):
                if visualized_batch[batch] == False:
                    continue
                prev_time      = self.time_history[batch][-1]
                curr_time      = self.current_time[batch].item()
                prev_veh_batt  = copy.deepcopy(self.vehicle_batt_history[batch])
                prev_loc_batt  = copy.deepcopy(self.loc_batt_history[batch])
                prev_down_locs = self.down_history[batch][-1]
                curr_veh_unavail_time = self.vehicle_unavail_time[batch].clamp(0.0).detach().clone()
                time_interval  = curr_time - prev_time
                dts = np.arange(OUTPUT_INTERVAL, time_interval, OUTPUT_INTERVAL).tolist()
                if len(dts) != 0:
                    if time_interval - dts[-1] <= OUTPUT_INTERVAL / 4:
                        dts[-1] = time_interval
                    else:
                        dts.append(time_interval)
                else:
                    dts.append(time_interval)
                for dt in dts:
                    ratio = dt / time_interval
                    curr_veh_batt = torch.tensor([
                        interpolate_line(prev_veh_batt[vehicle_id][-1], self.vehicle_curr_battery[batch][vehicle_id].item(), ratio)
                        for vehicle_id in range(self.num_vehicles)
                    ]) # [num_vehicles]
                    curr_loc_batt = torch.tensor([
                        interpolate_line(prev_loc_batt[loc_id][-1], self.loc_curr_battery[batch][loc_id].item(), ratio) 
                        for loc_id in range(self.num_locs)
                    ]) # [num_locs]
                    curr_down_locs = interpolate_line(prev_down_locs, ((self.loc_curr_battery[batch] - self.loc_min_battery) <= 0.0).sum().item(), ratio) # [1]
                    veh_unavail_time = curr_veh_unavail_time + (time_interval - dt)
                    self.visualize_state(batch, prev_time + dt, curr_veh_batt, curr_loc_batt, curr_down_locs, veh_unavail_time)

    def visualize_state(self, 
                        batch: int, 
                        curr_time: float,
                        curr_veh_batt: torch.FloatTensor,
                        curr_loc_batt: torch.FloatTensor,
                        curr_down_locs: float,
                        veh_unavail_time: torch.FloatTensor):
        #-----------------
        # battery history
        #-----------------
        self.time_history[batch].append(curr_time)
        for vehicle_id in range(self.num_vehicles):
            self.vehicle_batt_history[batch][vehicle_id].append(curr_veh_batt[vehicle_id].item())
        for loc_id in range(self.num_locs):
            self.loc_batt_history[batch][loc_id].append(curr_loc_batt[loc_id].item())
        self.down_history[batch].append(curr_down_locs)

        #---------------
        # visualziation
        #---------------
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(ncols=2, nrows=3, width_ratios=[1, 1.5])
        ax = fig.add_subplot(gs[:, 1])
        # current state
        loc_battery = torch2numpy(curr_loc_batt)         # [num_locs]
        vehicle_battery = torch2numpy(curr_veh_batt) # [num_vehicles]
        loc_cap = torch2numpy(self.loc_cap[batch])                      # [num_locs]
        loc_coords = torch2numpy(self.loc_coords[batch])                # [num_locs x coord_dim]
        depot_coords = torch2numpy(self.depot_coords[batch])            # [num_depots x coord_dim]
        coords = np.concatenate((loc_coords, depot_coords), 0)          # [num_nodes x coord_dim]
        vehicle_cap = torch2numpy(self.vehicle_cap[batch])              # [num_vehicles]
        x_loc = loc_coords[:, 0]; y_loc = loc_coords[:, 1]
        x_depot = depot_coords[:, 0]; y_depot = depot_coords[:, 1]
        # visualize nodes
        for id in range(self.num_locs):
            ratio = loc_battery[id] / loc_cap[id]
            add_base(x_loc[id], y_loc[id], ratio, ax)
        ax.scatter(x_depot, y_depot, marker="*", c="black", s=200, zorder=3)
        # visualize vehicles
        cmap = get_cmap(self.num_vehicles)
        for vehicle_id in range(self.num_vehicles):
            ratio = vehicle_battery[vehicle_id] / vehicle_cap[vehicle_id]
            vehicle_phase = self.vehicle_phase[batch][vehicle_id]
            vehicle_position_id = self.vehicle_position_id[batch][vehicle_id]
            if vehicle_phase != self.phase_id["move"]:
                vehicle_x = coords[vehicle_position_id, 0]
                vehicle_y = coords[vehicle_position_id, 1]
                add_vehicle(vehicle_x, vehicle_y, ratio, vehicle_battery[vehicle_id], cmap(vehicle_id), ax)
            else:
                vehicle_position_id_prev = self.vehicle_position_id_prev[batch][vehicle_id]
                speed = self.speed[batch]
                start = coords[vehicle_position_id_prev, :]
                end   = coords[vehicle_position_id, :]
                distance = np.linalg.norm(start - end)
                curr_position = interpolate_line(start, end, (1.0 - speed * veh_unavail_time[vehicle_id] / distance).item())
                ax.plot([start[0], curr_position[0]], [start[1], curr_position[1]], zorder=0, linestyle="-", color=cmap(vehicle_id))         # passed path
                ax.plot([curr_position[0], end[0]], [curr_position[1], end[1]], zorder=0, alpha=0.5, linestyle="--", color=cmap(vehicle_id)) # remaining path
                add_vehicle(curr_position[0], curr_position[1], ratio, vehicle_battery[vehicle_id], cmap(vehicle_id), ax)
        ax.set_title(f"current time = {curr_time:.3f} h", y=-0.05, fontsize=18)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
        ax.set_aspect(1)

        #----------------------------
        # add history plot until now
        #----------------------------
        time_horizon = self.time_horizon.cpu().item()
        max_veh_batt = torch.ceil(self.vehicle_cap[batch].max() / 10).cpu().item() * 10
        max_loc_batt = torch.ceil(self.loc_cap[batch].max() / 10).cpu().item() * 10
        max_num_locs = math.ceil(self.num_locs / 10) * 10
        # EV battery history
        ax_ev = fig.add_subplot(gs[0, 0])
        for vehicle_id in range(self.num_vehicles):
            ax_ev.plot(self.time_history[batch], list(self.vehicle_batt_history[batch][vehicle_id]), alpha=0.7, color=cmap(vehicle_id))
        ax_ev.set_xlim(0, time_horizon)
        ax_ev.set_ylim(0, max_veh_batt)
        ax_ev.get_xaxis().set_visible(False)
        ax_ev.axvline(x=self.time_history[batch][-1], ymin=-1.2, ymax=1, c="black", lw=1.5, zorder=0, clip_on=False)
        ax_ev.set_ylabel("EV battery (kWh)", fontsize=18)
        # Base station battery history
        ax_base = fig.add_subplot(gs[1, 0])
        for loc_id in range(self.num_locs):
            ax_base.plot(self.time_history[batch], list(self.loc_batt_history[batch][loc_id]), alpha=0.7)
        ax_base.set_xlim(0, time_horizon)
        ax_base.set_ylim(0, max_loc_batt)
        ax_base.get_xaxis().set_visible(False)
        ax_base.axvline(x=self.time_history[batch][-1], ymin=-1.2, ymax=1, c="black", lw=1.5, zorder=0, clip_on=False)
        ax_base.set_ylabel("Base station battery (kWh)", fontsize=18)
        # Num. of downed base stations
        ax_down = fig.add_subplot(gs[2, 0])
        ax_down.plot(self.time_history[batch], self.down_history[batch])
        ax_down.set_xlim(0, time_horizon)
        ax_down.set_ylim(0, max_num_locs)
        ax_down.axvline(x=self.time_history[batch][-1], ymin=0, ymax=1, c="black", lw=1.5, zorder=0, clip_on=False)
        ax_down.set_xlabel("Time (h)", fontsize=18)
        ax_down.set_ylabel("# downed base stations", fontsize=18)

        #---------------
        # save an image
        #---------------
        fig.subplots_adjust(left=0.03, right=1, bottom=0.05, top=0.98, wspace=0.05)
        fname = f"{self.fname}-{batch}/png/tour_state{self.episode_step}.png"
        os.makedirs(f"{self.fname}-{batch}/png", exist_ok=True)
        plt.savefig(fname, dpi=DPI)
        plt.close()
        self.episode_step += 1

    def output_batt_history(self):
        batch = 0
        fig = plt.figure(figsize=(10, 30))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        for vehicle_id in range(self.num_vehicles):
            ax1.plot(self.time_history[batch], list(self.vehicle_batt_history[batch][vehicle_id]))
        for loc_id in range(self.num_locs):
            ax2.plot(self.time_history[batch], list(self.loc_batt_history[batch][loc_id]))
        ax3.plot(self.time_history[batch], self.down_history[batch])
        ax1.set_xlabel("Time (h)")
        ax1.set_ylabel("EVs' battery (KW)")
        ax2.set_xlabel("Time (h)")
        ax2.set_ylabel("Base stations' battery (KW)")
        ax3.set_xlabel("Time (h)")
        ax3.set_ylabel("Number of downed base stations")
        plt.savefig(f"{self.fname}-{batch}/batt_history.png", dpi=DPI)
        # save raw data
        hisotry_data = {
            "time": self.time_history[batch],
            "veh_batt": self.vehicle_batt_history[batch],
            "loc_batt": self.loc_batt_history[batch],
            "down_loc": self.down_history[batch]
        }
        os.makedirs(f"{self.fname}-sample{batch}", exist_ok=True)
        with open(f"{self.fname}-sample{batch}/history_data.pkl", "wb") as f:
            pickle.dump(hisotry_data, f)
    
    def output_gif(self):
        for batch in range(1):
            anim_type = "mp4"
            out_fname = f"{self.fname}-{batch}/EVRoute.{anim_type}"
            seq_fname = f"{self.fname}-{batch}/png/tour_state%d.png"
            output_animation(out_fname, seq_fname, anim_type)

def torch2numpy(tensor: torch.Tensor):
    return tensor.cpu().detach().numpy().copy()

def add_base(x, y, ratio, ax):
    width = 0.01 * 1.5
    height = 0.015 * 1.5
    height_mod = ratio * height
    if ratio > 0.5:
        battery_color = "limegreen"
    elif ratio > 0.3:
        battery_color = "gold"
    else:
        battery_color = "red"
    ec = "red" if ratio < 1e-9 else "black"

    frame = patches.Rectangle(xy=(x-width/2, y-height/2), width=width, height=height, fill=False, ec=ec)
    battery = patches.Rectangle(xy=(x-width/2, y-height/2), width=width, height=height_mod, facecolor=battery_color, linewidth=.5, ec="black")
    ax.add_patch(battery)
    ax.add_patch(frame) 

def add_vehicle(x, y, ratio, batt, color, ax):
    offst = 0.03
    BATT_OFFSET = 0.025
    # vehicle_battery
    width = 0.015 * 1.2
    height = 0.01 * 1.2
    width_mod = ratio * width
    ec = "red" if ratio < 1e-9 else "black"
    frame = patches.Rectangle(xy=(x-width/2, y-height/2+offst+BATT_OFFSET), width=width, height=height, fill=False, ec=ec)
    battery = patches.Rectangle(xy=(x-width/2, y-height/2+offst+BATT_OFFSET), width=width_mod, height=height, facecolor=color, linewidth=.5, ec="black")
    ax.add_patch(battery)
    ax.add_patch(frame)

    # add remaining battery
    ax.text(x-0.02, y+0.07, f"{batt: .1f}", fontsize=10)
    
    # vehicle
    original_img = plt.imread("images/ev_image.png")
    vehicle_img = np.where(original_img == (1., 1., 1., 1.), (color[0], color[1], color[2], color[3]), original_img)
    vehicle_img = OffsetImage(vehicle_img, zoom=0.25)
    ab = AnnotationBbox(vehicle_img, (x, y+offst), xycoords='data', frameon=False)
    ax.add_artist(ab)

def get_cmap(num_colors: int):
    if num_colors <= 10:
        cm_name = "tab10"
    elif num_colors <= 20:
        cm_name = "tab20"
    else:
        assert False
    return cm.get_cmap(cm_name)

def output_animation(out_fname, seq_fname, type="gif"):
    if type == "gif":
        cmd = f"ffmpeg -r {FPS} -i {seq_fname} -r {FPS} {out_fname}"
    else:
        cmd = f"ffmpeg -r {FPS} -i {seq_fname} -vcodec libx264 -pix_fmt yuv420p -r {FPS} {out_fname}"
    subprocess.call(cmd, shell=True)

def interpolate_line(start, end, ratio):
    return ratio * (end - start) + start