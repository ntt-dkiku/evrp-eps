LARGE_VALUE = int(1e+6)
BIT_LARGE_VALUE = int(1e+4)

import os
import matplotlib.pyplot as plt
import collections
import math
import numpy as np
import torch
from scipy.spatial import distance
from ortools.sat.python import cp_model
from typing import List, Any, Tuple, Union

INFINITY = int(1e+14)

class TimeSpaceNetwork():
    def __init__(self, 
                 num_nodes: int,
                 T: int,
                 dt: float,
                 veh_speed: int,
                 distance_matrix: int,
                 max_traversal_step: int = 1) -> None:
        self.num_nodes = num_nodes
        self.T = T
        self.veh_speed = veh_speed
        self.distance_matrix = distance_matrix
        # 0 -> invalid, 1 -> valid
        self.valid_nodes = np.ones((num_nodes, T)) # all nodes are valid here
        self.valid_arcs  = np.zeros((num_nodes, num_nodes, T, T), dtype='int16')
        # remove arcs that cannot be reached within the time (traversal_steps * dt)
        for t1 in range(T):
            for t2 in range(t1+1, T):
                self.valid_arcs[:, :, t1, t2] = (distance_matrix <= veh_speed * (t2 - t1) * dt)
        # remove arcs that traversal more than (max_traversal_step+1) steps:
        # EVs alway reach a node at the earliest
        for t1 in range(T):
            for t2 in range(t1+1, T-max_traversal_step):
                for t3 in range(t2+max_traversal_step, T):
                    self.valid_arcs[:, :, t1, t3] = np.maximum(self.valid_arcs[:, :, t1, t3] - self.valid_arcs[:, :, t1, t2], 0)
        # remove duplicated stay arcs:
        # stay arcs that traverse more than 2 steps is not needed as it can be represented by two stay arcs that traverse only 1 step instead
        for t1 in range(T):
            for t2 in range(t1+2, T):
                np.fill_diagonal(self.valid_arcs[:, :, t1, t2], 0)

        # store valid nodes & arcs
        self.nodes = [(node_id, t) for node_id, t in zip(*np.where(self.valid_nodes))]
        self.arcs = [(from_node_id, to_node_id, from_time, to_time) for from_node_id, to_node_id, from_time, to_time in zip(*np.where(self.valid_arcs))]

    def inflow_arcs(self, to_node_id: int, to_t: int) -> List[Tuple[int, int, int, int]]:
        from_nodes = np.where(self.valid_arcs[:, to_node_id, :, to_t])
        inflows = [(from_node_id, to_node_id, from_t, to_t) for from_node_id, from_t in zip(*from_nodes)]
        return inflows

    def outflow_arcs(self, from_node_id: int, from_t: int) -> List[Tuple[int, int, int, int]]:
        to_nodes = np.where(self.valid_arcs[from_node_id, :, from_t, :])
        outflows = [(from_node_id, to_node_id, from_t, to_t) for to_node_id, to_t in zip(*to_nodes)]
        return outflows

    def stay_arcs(self, to_node_id: int, to_t: int) -> Union[Tuple[int, int, int, int], None]:
        if to_t > 0:
            return (to_node_id, to_node_id, to_t-1, to_t)
        else:
            return None

    def arriving_arcs(self, arriving_time: int) -> List[Tuple[int, int, int, int]]:
        from_nodes = np.where(self.valid_arcs[:, :, :, arriving_time])
        arrivings = [(from_id, to_id, from_time, arriving_time) for from_id, to_id, from_time in zip(*from_nodes)]
        return arrivings

    def time_slice(self, t: int) -> List[Tuple[int, int]]:
        return [(id[0], t) for id in zip(*np.where(self.valid_nodes[:, t]))]

    def arc_distance(self, arc: Tuple[int, int, int, int]) -> int:
        return self.distance_matrix[arc[0]][arc[1]]

    # def disp(self):
    #     print(self.nodes)
    #     print(self.distance_matrix)
    #     #print(self.arcs_array)
    #     block = np.block([[self.valid_arcs[t1, t2] for t2 in self.time_index] for t1 in self.time_index])
    #     print(block)

    def visualize(self, outputdir: str) -> None:
        pass
        os.makedirs(outputdir, exist_ok=True)
        # DEBUG用 arc, nodeの可視化
        self.plot_node_arc(self.nodes, self.arcs, outputdir)
        #self.plot_node_arc(self.time_slice(1), [])
        #self.plot_node_arc([], self.inflow_arcs(2, 1))
        #self.plot_node_arc([], self.outflow_arcs(1, 1))
        #self.plot_node_arc([], self.stay_arcs(2,1))

    def plot_node_arc(self, 
                      nodes: List[Tuple[int, int]], 
                      arcs: List[Tuple[int, int, int, int]], 
                      outputdir: str = None) -> None:
        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot()
        for node in nodes:
            ax.scatter(node[1], node[0], s=128, color='red')
        for arc in arcs:
            ax.annotate("", xy=[arc[3], arc[1]], xytext=[arc[2], arc[0]],
                        arrowprops=dict(shrink=0, width=0.2, headwidth=8, headlength=6, connectionstyle='arc3',
                                        facecolor='gray', edgecolor='gray', alpha=0.4))
        # ax.set_xticks(self.T)
        # ax.set_yticks(range(0, len(self.distance_matrix)))
        # ax.set_xlim(self.time_index[0], self.time_index[-1])
        # ax.set_ylim(-len(self.distance_matrix)*0.05, (len(self.distance_matrix)-1)*1.05)
        if outputdir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(outputdir, 'TSN.png'))

def flatten_list(l: list) -> List[Any]:
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten_list(el)
        else:
            yield el

class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables, limit) -> None:
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solution_limit = limit

    def on_solution_callback(self):
        self.__solution_count += 1
        # for v in self.__variables:
        #     print(f'{v}={self.Value(v)}')
        # print()
        if self.__solution_count >= self.__solution_limit:
            print(f'Stop search after {self.__solution_limit} solutions')
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count

class CP4TSN():
    def __init__(self,
                 time_horizon: int = 12,
                 dt: float = 0.5,
                 vehicle_speed: float = 41,
                 loss_coef: int = 1000,
                 loc_pre_time: float = 0.5,
                 loc_post_time: float = 0.5,
                 depot_pre_time: float = 0.17,
                 depot_post_time: float = 0.17,
                 ensure_minimum_charge: bool = True,
                 ensure_minimum_supply: bool = True,
                 random_seed: int = 1234,
                 num_search_workers: int = 4,
                 log_search_progress: bool = False,
                 limit_type: str = None,
                 time_limit: float = 60.0,
                 solution_limit: int = 10):
        """
        Parameters
        ----------

        """
        self.time_horizon = time_horizon
        self.dt = dt
        self.T  = int(time_horizon / dt) + 1
        self.vehicle_speed = vehicle_speed
        self.loss_coef = loss_coef
        self.loc_pre_time = loc_pre_time
        self.loc_post_time = loc_post_time
        self.depot_pre_time = depot_pre_time
        self.depot_post_time = depot_post_time
        self.ensure_minimum_charge = ensure_minimum_charge
        self.ensure_minimum_supply = ensure_minimum_supply
        self.loc_surplus_pre_time = loc_pre_time - dt * (math.ceil(loc_pre_time / dt) - 1)
        self.loc_surplus_post_time = loc_post_time - dt * (math.ceil(loc_post_time / dt) - 1)
        self.depot_surplus_pre_time = depot_pre_time - dt * (math.ceil(depot_pre_time / dt) - 1)
        self.depot_surplus_post_time = depot_post_time - dt * (math.ceil(depot_post_time / dt) - 1)

        self.random_seed = random_seed
        self.num_search_workers = num_search_workers
        self.log_search_progress = log_search_progress
        self.limit_type = limit_type
        # NOTE: time_limit could make the results unrepreducible even if random_seed is set 
        # because the calculation time could differ in each run, resulting in different numbers of found solution
        self.time_limit = time_limit
        self.solution_limit = solution_limit

    def solve(self, input: dict, log_fname: str = None):
        """
        Paramters
        ---------

        Returns
        -------

        """
        # convert input feature
        self.set_input(input)

        # deffine Time Space Network
        self.tsn = TimeSpaceNetwork(self.num_nodes, self.T, self.dt, self.normalized_veh_speed, self.distance_matrix)

        # define a model
        print("defining a model...", end="")
        model = cp_model.CpModel()
        variables = self.add_variables(model)
        self.add_constraints(model, variables)
        self.add_objectives(model, variables, self.loss_coef)
        print("done")

        # validate the model
        validate_res = model.Validate()
        if validate_res != "":
            print(validate_res)

        # solve TSN with the CP-SAT solver
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = self.random_seed
        solver.parameters.num_search_workers = self.num_search_workers
        solver.log_search_progress = self.log_search_progress
        if self.limit_type == "time":
            solver.parameters.max_time_in_seconds = self.time_limit
            status = solver.Solve(model)
        elif self.limit_type == "solution_count":
            solver.parameters.num_search_workers = 1 # enumerating all solutions does not work in parallel
            solver.parameters.enumerate_all_solutions = True
            variable_list = []
            for variable in variables.values():
                if isinstance(variable, list):
                    variable_list += list(flatten_list(variable))
                elif isinstance(variable, dict):
                    variable_list += list(variable.values())
                else:
                    variable_list += [variable]
            solution_printer = VarArraySolutionPrinterWithLimit(variable_list, self.solution_limit)
            status = solver.Solve(model, solution_printer)
        else:
            status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            print("The optimal solution found!!!")
        elif status == cp_model.FEASIBLE:
            print("A feasible solution found!")
        else:
            print("No solution found :(")

        # if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        #     print([solver.Value(variables["num_down_loc"][t]) for t in range(self.T)])
        #     print([solver.Value(variables["travel_distance"][veh_id]) for veh_id in range(self.num_vehicles)])
        #     for veh_id in range(self.num_vehicles):
        #         print(f"EV{veh_id}")
        #         route = [arc for arc in self.tsn.arcs if solver.Value(variables["x"][veh_id, arc]) == 1]
        #         route.sort(key=lambda a: a[2])
        #         for arc in route:
        #             print(f"(time:{arc[2]}->{arc[3]}) station:{arc[0]}-> {arc[1]}")
        #     print(solver.Value(variables["loss1"]) / (LARGE_VALUE * BIT_LARGE_VALUE), solver.Value(variables["loss2"]) / (LARGE_VALUE * BIT_LARGE_VALUE))
        # print(f"Status: {solver.StatusName(status)}")
        # print(solver.SolutionInfo())
        # print(solver.ResponseStats())
        
        route = [[] for _ in range(self.num_vehicles)]
        for veh_id in range(self.num_vehicles):
            arcs = [arc for arc in self.tsn.arcs if solver.Value(variables["x"][veh_id, arc]) == 1]
            arcs.sort(key=lambda a: a[2])
            for arc in arcs:
                route[veh_id].append((arc[2], arc[3], arc[0], arc[1]))
        return {
            "route": route,
            "total_route_length": sum(solver.Value(variables["travel_distance"][veh_id]) for veh_id in range(self.num_vehicles)) / LARGE_VALUE,
            "num_down_locs": [solver.Value(variables["num_down_loc"][t]) for t in range(self.T)],
            "objective_value": solver.ObjectiveValue() / (LARGE_VALUE * BIT_LARGE_VALUE)
        }
    
    def set_input(self, input: dict):
        # locations
        self.loc_cap = (input["loc_cap"] * LARGE_VALUE).to(torch.long).tolist() # [num_locs]
        self.loc_consump_rate = (input["loc_consump_rate"] * LARGE_VALUE).to(torch.long).tolist() # [num_locs]
        self.loc_init_batt = (input["loc_cap"] * LARGE_VALUE).to(torch.long).tolist() # [num_locs]
        self.loc_min_batt = 0

        # depots
        self.depot_discharge_rate = (input["depot_discharge_rate"] * LARGE_VALUE).to(torch.long).tolist() # [num_depots]
        
        # EVs
        self.veh_cap = (input["vehicle_cap"] * LARGE_VALUE).to(torch.long).tolist() # [num_vehicles]
        self.veh_init_batt = (input["vehicle_cap"] * LARGE_VALUE).to(torch.long).tolist() # [num_vehicles]
        self.veh_discharge_rate = (input["vehicle_discharge_rate"] * LARGE_VALUE).to(torch.long).tolist()
        self.veh_consump_rate = input["vehicle_consump_rate"].tolist()
        self.veh_init_position_id = input["vehicle_initial_position_id"].tolist() # [num_vehicles]
        self.veh_min_batt = 0

        # distance_matrix
        self.loc_coords = input["loc_coords"].detach().numpy().copy() # [num_locs, coord_dim]
        self.depot_coords = input["depot_coords"].detach().numpy().copy() # [num_depots, coord_dim]
        self.node_coords = np.concatenate((self.loc_coords, self.depot_coords), 0) # [num_nodes, coord_dim]
        self.distance_matrix = (distance.cdist(self.node_coords, self.node_coords) * LARGE_VALUE).astype(np.long)

        # parameters
        self.num_locs = len(self.loc_cap)
        self.num_depots = len(self.depot_discharge_rate)
        self.num_nodes = self.num_locs + self.num_depots
        self.num_vehicles = len(self.veh_cap)
        self.grid_scale = input["grid_scale"]
        self.normalized_veh_speed = int(self.vehicle_speed / self.grid_scale * LARGE_VALUE)

    def add_variables(self, model):
        """
        Parameters
        ----------

        Returns
        -------
        """
        variables = {}
        self.add_batt_variables(model, variables)
        self.add_route_variables(model, variables)
        self.add_objective_variables(model, variables)
        return variables

    def add_batt_variables(self, model, var):
        # for locations
        var["loc_batt"] = [[model.NewIntVar(0, self.loc_cap[i], f"loc{i}_t{t}_batt") for t in range(self.T)] for i in range(self.num_locs)]
        var["loc_slack"] = [[model.NewIntVar(0, self.loc_min_batt+1, f"loc{i}_t{t}_slack") for t in range(self.T)] for i in range(self.num_locs)]
        var["enable_slack"] = [[model.NewBoolVar(f"loc{i}_t{t}_enable_slack") for t in range(self.T)] for i in range(self.num_locs)]
        var["loc_is_down"] = [[model.NewBoolVar(f"loc{i}_t{t}_is_down") for t in range(self.T)] for i in range(self.num_locs)]
        var["loc_is_full"] = [[model.NewBoolVar(f"loc{i}_t{t}_is_full") for t in range(self.T)] for i in range(self.num_locs)]
        var["loc_is_normal"] = [[model.NewBoolVar(f"loc{i}_t{t}_is_normal") for t in range(self.T)] for i in range(self.num_locs)]
        var["loc_charge_amount"] = [[model.NewIntVar(0, self.loc_cap[i], f"loc{i}_t{t}_charge_amount") for t in range(self.T)] for i in range(self.num_locs)]
        # for EVs
        var["veh_batt"]           = [[model.NewIntVar(0, self.veh_cap[k], f"veh{k}_t{t}_batt") for t in range(self.T)] for k in range(self.num_vehicles)]
        var["veh_charge_amount"]  = [[model.NewIntVar(0, self.veh_cap[k], f"veh{k}_t{t}_charge_amount") for t in range(self.T)] for k in range(self.num_vehicles)]
        var["veh_is_discharging"] = [[model.NewBoolVar(f"veh{k}_t{t}_is_charging") for t in range(self.T)] for k in range(self.num_vehicles)]

    def add_route_variables(self, model, var):
        var["x"] = {(k, arc): model.NewBoolVar(f"x_veh{k}_arc{arc}") for k in range(self.num_vehicles) for arc in self.tsn.arcs}
        var["z"] = [[[model.NewBoolVar(f"z_veh{k}_node{n}_t{t}") for t in range(self.T)] for n in range(self.num_nodes)] for k in range(self.num_vehicles)]
        var["loc_is_down2"] = [[model.NewBoolVar(f"loc{i}_t{t}_is_down2") for t in range(self.T)] for i in range(self.num_locs)]
        var["veh_prepare_at_loc"]  = [[[model.NewBoolVar(f"veh{k}_loc{i}_t{t}_prepare_at_loc") for t in range(self.T)] for i in range(self.num_locs)] for k in range(self.num_vehicles)]
        var["veh_prepare_at_loc2"] = [[[model.NewBoolVar(f"veh{k}_loc{i}_t{t}_prepare_at_loc2") for t in range(self.T)] for i in range(self.num_locs)] for k in range(self.num_vehicles)] 
        var["veh_cleanup_at_loc"]  = [[[model.NewBoolVar(f"veh{k}_loc{i}_t{t}_cleanup_at_loc") for t in range(self.T)] for i in range(self.num_locs)] for k in range(self.num_vehicles)]
        var["veh_cleanup_at_loc2"] = [[[model.NewBoolVar(f"veh{k}_loc{i}_t{t}_cleanup_at_loc2") for t in range(self.T)] for i in range(self.num_locs)] for k in range(self.num_vehicles)]
        var["veh_prepare_at_depot"] = [[[model.NewBoolVar(f"veh{k}_depot{j}_t{t}_prepare_at_depot") for t in range(self.T)] for j in range(self.num_depots)] for k in range(self.num_vehicles)]
        var["veh_cleanup_at_depot"] = [[[model.NewBoolVar(f"veh{k}_depot{j}_t{t}_cleanup_at_depot") for t in range(self.T)] for j in range(self.num_depots)] for k in range(self.num_vehicles)]
        if self.ensure_minimum_supply:
            var["loc_supply_notenough"] = [[[model.NewBoolVar(f"not_enough_veh{k}_loc{i}_t{t}") for t in range(self.T)] for i in range(self.num_locs)] for k in range(self.num_vehicles)]
            var["loc_suuply_notenough_notcleanup"] = [[[model.NewBoolVar(f"notenough_notcleanup_veh{k}_loc{i}_t{t}") for t in range(self.T)] for i in range(self.num_locs)] for k in range(self.num_vehicles)]
        if self.ensure_minimum_charge:
            var["veh_charge_notenough"] = [[[model.NewBoolVar(f"not_enough_veh{k}_depot{j}_t{t}") for t in range(self.T)] for j in range(self.num_depots)] for k in range(self.num_vehicles)]
            var["veh_charge_notenough_notcleanup"] = [[[model.NewBoolVar(f"notenough_notcleanup_veh{k}_depot{j}_t{t}") for t in range(self.T)] for j in range(self.num_depots)] for k in range(self.num_vehicles)]

    def add_objective_variables(self, model, var):
        var["num_down_loc"] = [model.NewIntVar(0, self.num_locs, f"num_down_loc_t{t}") for t in range(self.T)]
        var["travel_distance"] = [model.NewIntVar(0, INFINITY, f"veh{k}_travel_distance") for k in range(self.num_vehicles)]
        var["loss1"] = model.NewIntVar(0, INFINITY, "loss1")
        var["loss2"] = model.NewIntVar(0, INFINITY, "loss2")

    def add_constraints(self, model, var):
        self.battery_init_lowerbound(model, var)
        self.ensure_route_continuity(model, var)
        self.forbit_multi_veh_at_same_node(model, var)
        self.define_batt_behavior(model, var)
        self.add_objective_constraints(model, var)

    def battery_init_lowerbound(self, model, var):
        # for EVs
        for veh_id in range(self.num_vehicles):
            model.Add(var["veh_batt"][veh_id][0] == self.veh_init_batt[veh_id])
            for t in range(self.T):
                model.Add(var["veh_is_discharging"][veh_id][t] == sum(var["z"][veh_id][loc_id][t] for loc_id in range(self.num_locs)))
                model.Add(var["veh_batt"][veh_id][t] >= self.veh_min_batt).OnlyEnforceIf(var["veh_is_discharging"][veh_id][t])
        # for locations
        for loc_id in range(self.num_locs):
            model.Add(var["loc_batt"][loc_id][0] == self.loc_init_batt[loc_id])
            for t in range(self.T):
                # implement enable_slack
                model.Add(var["loc_batt"][loc_id][t] < self.loc_min_batt).OnlyEnforceIf(var["enable_slack"][loc_id][t])
                model.Add(var["loc_batt"][loc_id][t] >= self.loc_min_batt).OnlyEnforceIf(var["enable_slack"][loc_id][t].Not())
                # implement loc_slack
                model.Add(var["loc_slack"][loc_id][t] >= 1).OnlyEnforceIf(var["enable_slack"][loc_id][t])
                model.Add(var["loc_slack"][loc_id][t] == 0).OnlyEnforceIf(var["enable_slack"][loc_id][t].Not())
                # add a constraint
                model.Add(var["loc_batt"][loc_id][t] + var["loc_slack"][loc_id][t] >= self.loc_min_batt)

    def ensure_route_continuity(self, model, var):
        """
        Parameters
        ----------

        """
        for veh_id in range(self.num_vehicles):
            # set initial position 
            # outflow arcs of the first node for a vehcile is 1
            outflow_arcs_from_init_depot = self.tsn.outflow_arcs(self.veh_init_position_id[veh_id], 0)
            model.Add(sum(var["x"][veh_id, arc] for arc in outflow_arcs_from_init_depot) == 1)
            
            # 
            for node_id in range(self.num_nodes):
                if node_id in self.veh_init_position_id:
                    continue
                outflow_arcs = self.tsn.outflow_arcs(node_id, 0)
                for arc in outflow_arcs:
                    model.Add(var["x"][veh_id, arc] == 0)

            # route continuity: the number outflow arcs shold equals to the number of inflow arcs in a node
            # To handle sparsified TSN, we use get_xxx_arcs function
            for t in range(1, self.T-1):
                for n in range(self.num_nodes):
                    inflow_arcs  = self.tsn.inflow_arcs(n, t)  # get valid inflow arcs
                    outflow_arcs = self.tsn.outflow_arcs(n, t) # get valid outflow arcs
                    model.Add(sum(var["x"][veh_id, arc] for arc in inflow_arcs) == sum(var["x"][veh_id, arc] for arc in outflow_arcs))

            # 
            for t in range(0, self.T):
                for n in range(self.num_nodes):
                    if t == 0:
                        model.Add(var["z"][veh_id][n][t] == 0)
                    else:
                        stay_arc = self.tsn.stay_arcs(n, t) # get stay arc
                        if stay_arc is not None:
                            model.Add(var["z"][veh_id][n][t] <= var["x"][veh_id, stay_arc])

    def forbit_multi_veh_at_same_node(self, model, var):
        for t in range(1, self.T):
            for n in range(self.num_nodes):
                inflow_arcs = self.tsn.inflow_arcs(n, t)
                model.Add(sum(var["x"][veh_id, arc] for arc in inflow_arcs for veh_id in range(self.num_vehicles)) <= 1)
  
    def define_batt_behavior(self, model, var):
        prepare_t = math.ceil(self.loc_pre_time / self.dt)
        for loc_id in range(self.num_locs):
            for veh_id in range(self.num_vehicles):
                for p in range(prepare_t):
                    model.Add(var["veh_prepare_at_loc"][veh_id][loc_id][p] == 0)
                    model.Add(var["veh_cleanup_at_loc"][veh_id][loc_id][self.T-p-1] == 0)
                model.Add(var["veh_prepare_at_loc2"][veh_id][loc_id][self.T-1] == 0)
                model.Add(var["veh_cleanup_at_loc2"][veh_id][loc_id][0] == 0)

        for t in range(self.T-1):
            prev_t = t
            curr_t = t + 1
            # for EVs
            arriving_arcs = self.tsn.arriving_arcs(curr_t)
            for veh_id in range(self.num_vehicles):
                # charging: depot -> vehcile
                model.Add(var["veh_charge_amount"][veh_id][curr_t] == sum([int(self.depot_discharge_rate[depot_offst_id] * self.dt) * var["z"][veh_id][depot_id][curr_t] for depot_offst_id, depot_id in enumerate(range(self.num_locs, self.num_nodes))])
                          - sum(int(self.veh_discharge_rate[veh_id] * self.depot_surplus_pre_time) * var["veh_prepare_at_depot"][veh_id][depot_offst_id][curr_t] for depot_offst_id in range(self.num_depots))   # TODO
                          - sum(int(self.veh_discharge_rate[veh_id] * self.depot_surplus_post_time) * var["veh_cleanup_at_depot"][veh_id][depot_offst_id][curr_t] for depot_offst_id in range(self.num_depots))) # TODO
                # EV's battery change
                model.Add(var["veh_batt"][veh_id][curr_t] == var["veh_batt"][veh_id][prev_t] 
                          - sum([int(self.veh_discharge_rate[veh_id] * self.dt) * var["z"][veh_id][i][curr_t] for i in range(self.num_locs)]) # discharge consumption 
                          - sum([int(self.veh_consump_rate[veh_id] * self.tsn.arc_distance(arc)) * var["x"][veh_id, arc] for arc in arriving_arcs])              # travel consumption
                          + var["veh_charge_amount"][veh_id][curr_t])                                                     # power charge form a charge station 
            
            # for locations
            for loc_id in range(self.num_locs):
                for veh_id in range(self.num_vehicles):
                    if prev_t + prepare_t < self.T: # if prepare time does not exceed the time horizon
                        # implement veh_prepare_at_loc & veh_cleanup_at_loc
                        diff_z_loc = var["z"][veh_id][loc_id][prev_t+prepare_t] - var["z"][veh_id][loc_id][prev_t]
                        model.Add(diff_z_loc == 1).OnlyEnforceIf(var["veh_prepare_at_loc"][veh_id][loc_id][prev_t+prepare_t])
                        model.Add(diff_z_loc <= 0).OnlyEnforceIf(var["veh_prepare_at_loc"][veh_id][loc_id][prev_t+prepare_t].Not())
                        model.Add(-diff_z_loc == 1).OnlyEnforceIf(var["veh_cleanup_at_loc"][veh_id][loc_id][prev_t])
                        model.Add(-diff_z_loc <= 0).OnlyEnforceIf(var["veh_cleanup_at_loc"][veh_id][loc_id][prev_t].Not())
                    # implement veh_prepare_at_loc & veh_cleanup_at_loc
                    diff_veh_prepare_at_loc = var["veh_prepare_at_loc"][veh_id][loc_id][prev_t] - var["veh_prepare_at_loc"][veh_id][loc_id][curr_t]
                    diff_veh_cleanup_at_loc = var["veh_cleanup_at_loc"][veh_id][loc_id][curr_t] - var["veh_cleanup_at_loc"][veh_id][loc_id][prev_t]
                    model.Add(diff_veh_prepare_at_loc == 1).OnlyEnforceIf(var["veh_prepare_at_loc2"][veh_id][loc_id][prev_t])
                    model.Add(diff_veh_prepare_at_loc <= 0).OnlyEnforceIf(var["veh_prepare_at_loc2"][veh_id][loc_id][prev_t].Not())
                    model.Add(diff_veh_cleanup_at_loc == 1).OnlyEnforceIf(var["veh_cleanup_at_loc2"][veh_id][loc_id][curr_t])
                    model.Add(diff_veh_cleanup_at_loc <= 0).OnlyEnforceIf(var["veh_cleanup_at_loc2"][veh_id][loc_id][curr_t].Not())
                if self.ensure_minimum_supply:
                    for veh_id in range(self.num_vehicles):
                        # implement loc_supply_not_enough
                        model.Add(self.loc_cap[loc_id] - var["loc_batt"][loc_id][prev_t] >  0).OnlyEnforceIf(var["loc_supply_notenough"][veh_id][loc_id][prev_t])
                        model.Add(self.loc_cap[loc_id] - var["loc_batt"][loc_id][prev_t] <= 0).OnlyEnforceIf(var["loc_supply_notenough"][veh_id][loc_id][prev_t].Not())
                        # implement loc_suuply_notenough_notcleanup
                        model.AddImplication(var["loc_suuply_notenough_notcleanup"][veh_id][loc_id][prev_t], var["loc_supply_notenough"][veh_id][loc_id][prev_t])
                        model.AddImplication(var["loc_suuply_notenough_notcleanup"][veh_id][loc_id][prev_t], var["veh_cleanup_at_loc"][veh_id][loc_id][prev_t].Not())
                        # add a constraint
                        model.Add(var["z"][veh_id][loc_id][curr_t] >= var["z"][veh_id][loc_id][prev_t]).OnlyEnforceIf(var["loc_suuply_notenough_notcleanup"][veh_id][loc_id][prev_t])

                # supplying: 
                model.Add(var["loc_charge_amount"][loc_id][curr_t] == int(self.veh_discharge_rate[veh_id] * self.dt) * sum([var["z"][veh_id_][loc_id][curr_t] - var["veh_prepare_at_loc"][veh_id_][loc_id][curr_t] - var["veh_cleanup_at_loc"][veh_id_][loc_id][curr_t] for veh_id_ in range(self.num_vehicles)])
                                                                      + int(self.veh_discharge_rate[veh_id] * (self.dt - self.loc_surplus_pre_time)) * sum(var["veh_prepare_at_loc2"][veh_id_][loc_id][curr_t] for veh_id_ in range(self.num_vehicles))
                                                                      + int(self.veh_discharge_rate[veh_id] * (self.dt - self.loc_surplus_post_time)) * sum(var["veh_cleanup_at_loc2"][veh_id_][loc_id][curr_t] for veh_id_ in range(self.num_vehicles)))
                # location's battery change
                model.Add(var["loc_batt"][loc_id][curr_t] == var["loc_batt"][loc_id][prev_t]
                            - (1 - var["loc_is_down"][loc_id][curr_t]) * int(self.loc_consump_rate[loc_id] * self.dt)
                            + var["loc_charge_amount"][loc_id][curr_t]
                            ).OnlyEnforceIf(var["loc_is_normal"][loc_id][curr_t])
                
                # clippling battery
                # implement loc_is_down & loc_is_full
                loc_curr_batt = var["loc_batt"][loc_id][prev_t] - int(self.loc_consump_rate[loc_id] * self.dt) + var["loc_charge_amount"][loc_id][curr_t]
                model.Add(loc_curr_batt <= 0).OnlyEnforceIf(var["loc_is_down"][loc_id][curr_t])
                model.Add(loc_curr_batt >  0).OnlyEnforceIf(var["loc_is_down"][loc_id][curr_t].Not())
                model.Add(loc_curr_batt >= self.loc_cap[loc_id]).OnlyEnforceIf(var["loc_is_full"][loc_id][curr_t])
                model.Add(loc_curr_batt <  self.loc_cap[loc_id]).OnlyEnforceIf(var["loc_is_full"][loc_id][curr_t].Not())
                # clip battery
                model.Add(var["loc_batt"][loc_id][curr_t] == 0).OnlyEnforceIf(var["loc_is_down"][loc_id][curr_t])
                model.Add(var["loc_batt"][loc_id][curr_t] == self.loc_cap[loc_id]).OnlyEnforceIf(var["loc_is_full"][loc_id][curr_t])

            # for depots
            for depot_offst_id, depot_id in enumerate(range(self.num_locs, self.num_nodes)):
                for veh_id in range(self.num_vehicles):
                    # implement veh_prepare_at_depot & veh_cleanup_at_depot
                    diff_z_depot = var["z"][veh_id][depot_id][curr_t] - var["z"][veh_id][depot_id][prev_t]
                    model.Add(diff_z_depot == 1).OnlyEnforceIf(var["veh_prepare_at_depot"][veh_id][depot_offst_id][curr_t])
                    model.Add(diff_z_depot <= 0).OnlyEnforceIf(var["veh_prepare_at_depot"][veh_id][depot_offst_id][curr_t].Not())
                    model.Add(-diff_z_depot == 1).OnlyEnforceIf(var["veh_cleanup_at_depot"][veh_id][depot_offst_id][prev_t])
                    model.Add(-diff_z_depot <= 0).OnlyEnforceIf(var["veh_cleanup_at_depot"][veh_id][depot_offst_id][prev_t].Not())
                if self.ensure_minimum_charge:
                    for veh_id in range(self.num_vehicles):
                        # implement veh_charge_notenough
                        model.Add(self.veh_cap[veh_id] - var["veh_batt"][veh_id][prev_t] >  0).OnlyEnforceIf(var["veh_charge_notenough"][veh_id][depot_offst_id][prev_t])
                        model.Add(self.veh_cap[veh_id] - var["veh_batt"][veh_id][prev_t] <= 0).OnlyEnforceIf(var["veh_charge_notenough"][veh_id][depot_offst_id][prev_t].Not())
                        # implement 
                        model.AddImplication(var["veh_charge_notenough_notcleanup"][veh_id][depot_offst_id][prev_t], var["veh_charge_notenough"][veh_id][depot_offst_id][prev_t])
                        model.AddImplication(var["veh_charge_notenough_notcleanup"][veh_id][depot_offst_id][prev_t], var["veh_cleanup_at_depot"][veh_id][depot_offst_id][prev_t].Not())
                        # add a constraint
                        model.Add(var["z"][veh_id][depot_id][curr_t] >= var["z"][veh_id][depot_id][prev_t]).OnlyEnforceIf(var["veh_charge_notenough_notcleanup"][veh_id][depot_offst_id][prev_t])

        for t in range(self.T):
            for loc_id in range(self.num_locs):
                model.AddBoolOr([var["loc_is_down"][loc_id][t], var["loc_is_full"][loc_id][t], var["loc_is_normal"][loc_id][t]]) # the sate of a location is down or ful or normal
                model.Add(var["loc_batt"][loc_id][t] <= 0).OnlyEnforceIf(var["loc_is_down2"][loc_id][t])
                model.Add(var["loc_batt"][loc_id][t] >= 1).OnlyEnforceIf(var["loc_is_down2"][loc_id][t].Not())

    def add_objective_constraints(self, model, var):
        # calculate the number of downed locations at each step
        for t in range(self.T):
            model.Add(var["num_down_loc"][t] == sum(var["loc_is_down2"][loc_id_][t] for loc_id_ in range(self.num_locs)))
        # calculate the total travel distance of each EV
        for veh_id in range(self.num_vehicles):
            model.Add(var["travel_distance"][veh_id] == sum(var["x"][veh_id, arc] * self.tsn.arc_distance(arc) for arc in self.tsn.arcs))

    def add_objectives(self, 
                       model, 
                       var: dict, 
                       loss_coef: int):
        # for travel distance
        avg_travel_distance = sum(var["travel_distance"][veh_id] for veh_id in range(self.num_vehicles)) * int(BIT_LARGE_VALUE / self.num_vehicles)
        # for downed locs
        down_rate = sum(var["num_down_loc"][t] for t in range(self.T)) * int(LARGE_VALUE / (self.num_locs * self.T)) * BIT_LARGE_VALUE
        model.Add(var["loss1"] == avg_travel_distance)
        model.Add(var["loss2"] == loss_coef * down_rate)
        # define objectives
        model.Minimize(avg_travel_distance + loss_coef * down_rate)