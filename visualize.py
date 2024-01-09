from utils.util import load_dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patches
from copy import copy

DPI = 150
VIS_OFFSET = 0.02
BATT_OFFSET = 0.02

def add_base(x, y, ratio, ax):
    width = 0.01
    height = 0.015
    height_mod = ratio * height
    if ratio > 0.5:
        battery_color = "limegreen"
    elif ratio > 0.3:
        battery_color = "gold"
    else:
        battery_color = "red"
    
    if ratio < 1e-9:
        ec = "red"
    else:
        ec = "black"

    frame = patches.Rectangle(xy=(x-width/2, y-height/2), width=width, height=height, fill=False, ec=ec)
    battery = patches.Rectangle(xy=(x-width/2, y-height/2), width=width, height=height_mod, facecolor=battery_color, linewidth=.5, ec="black")
    ax.add_patch(battery)
    ax.add_patch(frame)

def add_vehicle(x, y, ratio, color, ax, offst=0.0):
    # vehicle_battery
    # ratio = 0.4
    width = 0.015
    height = 0.01
    width_mod = ratio * width
    if ratio < 1e-9:
        ec = "red"
    else:
        ec = "black"
    frame = patches.Rectangle(xy=(x-width/2, y-height/2+offst+BATT_OFFSET), width=width, height=height, fill=False, ec=ec)
    battery = patches.Rectangle(xy=(x-width/2, y-height/2+offst+BATT_OFFSET), width=width_mod, height=height, facecolor=color, linewidth=.5, ec="black")
    ax.add_patch(battery)
    ax.add_patch(frame)

    # vehicle
    original_img = plt.imread("images/ev_image.png")
    vehicle_img = np.where(original_img == (1., 1., 1., 1.), (color[0], color[1], color[2], color[3]), original_img)
    vehicle_img = OffsetImage(vehicle_img, zoom=0.1)
    ab = AnnotationBbox(vehicle_img, (x, y+offst), xycoords='data', frameon=False)
    ax.add_artist(ab)


def visualize_tour(dataset_path, tour_path, save_dir, instance, anim_type):
    dataset = load_dataset(dataset_path)
    tours = load_dataset(tour_path)
    data = dataset[instance]
    tour = tours[instance]

    # node information
    loc_coords = data["loc_coords"] # [num_locs x coord_dim]
    depot_coords = data["depot_coords"] # [num_depots x coord_dim]
    coords = torch.cat((loc_coords, depot_coords), 0) # [num_nodes x coord_dim]
    x_loc = loc_coords[:, 0]; y_loc = loc_coords[:, 1]
    x_depot = depot_coords[:, 0]; y_depot = depot_coords[:, 1]

    num_locs      = len(x_loc)
    num_depots    = len(x_depot)
    num_vehicles  = len(tour)
    vehicle_steps = [0] * num_vehicles
    vehicle_travel_time    = np.zeros(num_vehicles)
    vehicle_charge_time    = np.zeros(num_vehicles)
    vehicle_unavail_time   = np.zeros(num_vehicles)
    estimated_unavail_time = np.zeros(num_vehicles)
    vehicle_phase = ["move" for _ in range(num_vehicles)]
    finished = ["end" for _ in range(num_vehicles)]
    vehicle_visit = np.zeros((num_vehicles, 2, 2)) # stores x_curr, y_curr, x_next, y_next
    vehicle_max_steps = [len(tour[vehicle_id]) for vehicle_id in range(num_vehicles)]
    loc_battery = data["loc_initial_battery"] # [num_locs]
    loc_cap = data["loc_cap"] # [num_locs]
    loc_consump_rate = data["loc_consump_rate"] # [num_locs]
    vehicle_discharge_rate = data["vehicle_discharge_rate"] # [num_vehicles]
    vehicle_position = np.zeros(num_vehicles).astype(int) # [num_vehicles]
    vehicle_battery = data["vehicle_cap"].clone() # [num_vehicles]
    vehicle_cap = data["vehicle_cap"].clone() # [num_vehicles]
    depot_discharge_rate = data["depot_discharge_rate"]
    curr_time = 0.0
    
    # select color map
    if num_vehicles <= 10:
        cm_name = "tab10"
    elif num_vehicles <= 20:
        cm_name = "tab20"
    else:
        assert False
    cmap = cm.get_cmap(cm_name)

    #----------------------------
    # visualize the initial step
    #----------------------------
    # initialize a fig instance
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # add locations & depots
    for id in range(num_locs):
        ratio = loc_battery[id] / loc_cap[id]
        add_base(x_loc[id], y_loc[id], ratio, ax)
    ax.scatter(x_depot, y_depot, marker="*", c="black", s=100, zorder=3)

    # initial vehicle assignment
    NODE_ID = 0; TRAVEL_TIME = 1; CHARGE_TIME = 2
    for i in range(num_vehicles):
        vehicle_steps[i] += 1
        vehicle_position[i] = tour[i][vehicle_steps[i]][NODE_ID]
        vehicle_visit[i, 0, 0] = coords[tour[i][vehicle_steps[i]-1][NODE_ID], 0] # x_curr
        vehicle_visit[i, 0, 1] = coords[tour[i][vehicle_steps[i]-1][NODE_ID], 1] # y_curr
        vehicle_visit[i, 1, 0] = coords[tour[i][vehicle_steps[i]][NODE_ID], 0] # x_next
        vehicle_visit[i, 1, 1] = coords[tour[i][vehicle_steps[i]][NODE_ID], 1] # y_next
        ax.plot(vehicle_visit[i, :, 0], vehicle_visit[i, :, 1], zorder=0, alpha=0.5, linestyle="--", color=cmap(i))
        vehicle_travel_time[i] = tour[i][vehicle_steps[i]][TRAVEL_TIME]
        vehicle_charge_time[i] = tour[i][vehicle_steps[i]][CHARGE_TIME]
        vehicle_unavail_time[i] = vehicle_travel_time[i]
        estimated_unavail_time[i] = vehicle_travel_time[i]
        # add a vehicle to image
        ratio = vehicle_battery[i] / vehicle_cap[i]
        add_vehicle(vehicle_visit[i, 0, 0], vehicle_visit[i, 0, 1], ratio, cmap(i), ax, VIS_OFFSET)

    # plt.savefig(f"{save_dir}/vis/png/tour_test.png")
    ax.set_title(f"current_time = {curr_time:.3f}")
    plt.xlim(-0.05, 1.05); plt.ylim(-0.05, 1.05)
    plt.savefig(f"{save_dir}/vis/png/tour_test0.png", dpi=DPI)
    plt.close()

    #-------------------------------
    # visualize the subseqent steps
    #-------------------------------
    total_steps = 1
    all_finished = False 
    while not all_finished:
        # select next vehicle
        next_vehicle_id = np.argmin(vehicle_unavail_time)
        i = next_vehicle_id

        # update time
        elapsed_time = vehicle_unavail_time[i].copy()
        curr_time += vehicle_unavail_time[i]
        vehicle_unavail_time -= vehicle_unavail_time[i]
        vehicle_unavail_time = vehicle_unavail_time.clip(0.0)

        # initialize a fig instance
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # add locations & depots
        charged_loc = []
        for vehicle_id in range(num_vehicles):
            loc_id = vehicle_position[vehicle_id]
            at_loc = loc_id < num_locs
            charging = vehicle_phase[vehicle_id] == "charge"
            if at_loc & charging:
                loc_battery[loc_id] += vehicle_discharge_rate[vehicle_id] * elapsed_time
                vehicle_battery[vehicle_id] -= vehicle_discharge_rate[vehicle_id] * elapsed_time
                vehicle_battery[vehicle_id] = vehicle_battery[vehicle_id].clip(0.0)
                ratio = loc_battery[loc_id] / loc_cap[loc_id]
                add_base(x_loc[loc_id], y_loc[loc_id], ratio, ax)
                charged_loc.append(loc_id)
            if (not at_loc) & charging:
                vehicle_battery[vehicle_id] += depot_discharge_rate[loc_id - num_locs] * elapsed_time

        for id in range(num_locs):
            if not (id in charged_loc):
                loc_battery[id] -= loc_consump_rate[id] * elapsed_time
                loc_battery[id] = loc_battery[id].clamp(0.0)
                ratio = loc_battery[id] / loc_cap[id]
                add_base(x_loc[id], y_loc[id], ratio, ax)
        # ax.scatter(x_loc, y_loc, marker="o", c="black", zorder=3)
        ax.scatter(x_depot, y_depot, marker="*", c="black", s=100, zorder=3)

        #-----------------------------------
        # visualization of selected vehicle
        #-----------------------------------
        # add the path
        if vehicle_phase[i] == "move":
            ax.plot(vehicle_visit[i, :, 0], vehicle_visit[i, :, 1], zorder=0, linestyle="-", color=cmap(i))
        # add selected vehicle to image
        ratio = vehicle_battery[i] / vehicle_cap[i]
        add_vehicle(vehicle_visit[i, 1, 0], vehicle_visit[i, 1, 1], ratio, cmap(i), ax, VIS_OFFSET)

        #---------------------------------
        # visualization of other vehicles
        #---------------------------------
        for k in range(num_vehicles):
            if k != i:
                x_st  = vehicle_visit[k, 0, 0]; y_st  = vehicle_visit[k, 0, 1]
                x_end = vehicle_visit[k, 1, 0]; y_end = vehicle_visit[k, 1, 1]
                if vehicle_phase[k] == "move":
                    progress = 1.0 - (vehicle_unavail_time[k] / estimated_unavail_time[k])
                    x_curr = progress * (x_end - x_st) + x_st
                    y_curr = progress * (y_end - y_st) + y_st
                    ax.plot([x_st, x_curr], [y_st, y_curr], zorder=0, linestyle="-", color=cmap(k))
                    ax.plot([x_curr, x_end], [y_curr, y_end], zorder=0, alpha=0.5, linestyle="--", color=cmap(k))
                    x_vehicle = x_curr; y_vehicle = y_curr
                    vis_offst = 0.0
                else:
                    x_vehicle = x_end; y_vehicle = y_end
                    vis_offst = VIS_OFFSET
                # add other vehicles to image
                ratio = vehicle_battery[k] / vehicle_cap[k]
                add_vehicle(x_vehicle, y_vehicle, ratio, cmap(k), ax, vis_offst)

        #--------------
        # update state
        #--------------
        if vehicle_phase[i] == "move":
            vehicle_unavail_time[i] = vehicle_charge_time[i].copy()
            estimated_unavail_time[i] = vehicle_charge_time[i].copy()
            if vehicle_steps[next_vehicle_id] >= vehicle_max_steps[next_vehicle_id] - 1:
                vehicle_phase[i] = "end"
                vehicle_unavail_time[i] = 1e+9
            else:
                vehicle_phase[i] = "charge"
        elif vehicle_phase[i] == "charge":
            vehicle_steps[next_vehicle_id] += 1
            vehicle_position[i] = tour[i][vehicle_steps[i]][NODE_ID]
            vehicle_travel_time[i] = tour[i][vehicle_steps[i]][TRAVEL_TIME]
            vehicle_charge_time[i] = tour[i][vehicle_steps[i]][CHARGE_TIME]
            vehicle_unavail_time[i]   = vehicle_travel_time[i].copy()
            estimated_unavail_time[i] = vehicle_travel_time[i].copy()
            vehicle_visit[i, 0, 0] = coords[tour[i][vehicle_steps[i]-1][NODE_ID], 0]
            vehicle_visit[i, 0, 1] = coords[tour[i][vehicle_steps[i]-1][NODE_ID], 1]
            vehicle_visit[i, 1, 0] = coords[tour[i][vehicle_steps[i]][NODE_ID], 0]
            vehicle_visit[i, 1, 1] = coords[tour[i][vehicle_steps[i]][NODE_ID], 1]
            vehicle_phase[i] = "move"

        if elapsed_time < 1e-9:
            plt.close()
            all_finished = not (vehicle_phase != finished)
            continue
        else:
            ax.set_title(f"current_time = {curr_time:.3f}")
            plt.xlim(-0.05, 1.05); plt.ylim(-0.05, 1.05)
            plt.savefig(f"{save_dir}/vis/png/tour_test{total_steps}.png", dpi=DPI)
            plt.close()
        # plt.savefig(f"{save_dir}/vis/png/tour_test.png")

        total_steps += 1
        all_finished = not (vehicle_phase != finished)

    # generate a gif from png files 
    gif_fig = plt.figure(figsize=(10, 10))
    pic_list = [f"{save_dir}/vis/png/tour_test{i}.png" for i in range(total_steps)]
    ims = []
    for i in range(len(pic_list)):
        im = Image.open(pic_list[i])
        ims.append([plt.imshow(im)])
    plt.axis("off")
    gif_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    gif = animation.ArtistAnimation(gif_fig, ims, interval=500, repeat_delay=5000)
    if anim_type == "gif":
        gif.save(f"{save_dir}/vis/test.gif", writer="pillow")
    else:
        gif.save(f"{save_dir}/vis/test.mp4", writer="ffmpeg")

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--tour_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--anim_type", type=str, default="gif")
    args = parser.parse_args()
    os.makedirs(f"{args.save_dir}/vis/png", exist_ok=True)
    visualize_tour(args.dataset_path, args.tour_path, args.save_dir, args.instance, args.anim_type)