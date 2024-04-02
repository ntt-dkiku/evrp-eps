# Electric Vehicle Routing for Emergency Power Supply: Towards Telecom Base Station Relief
This repo is the official implementation of Electric Vehicle Routing for Emergency Power Supply with Deep Reinforcement Learning (AAMAS 2024, extended abstract).
Please check also [the project page](https://ntt-dkiku.github.io/rl-evrpeps/).

<div><video autoplay loop controls src="https://github.com/ntt-dkiku/evrp-eps/assets/154794155/818b81f7-8e19-40ac-9934-70bc30f13e53"></video></div>

## Python Environment
We recommend using Docker to construct the python environment. You can use the [Dockerfile](./Dockerfile) in this repository. 
```
docker build -t evrp-eps/evrp-eps:1.0 .
``` 

## Usage
### 1. Generating synthetic data
First of all, we generate synthetic datasets for training/validation/evaluation. We recommend more than 12.8M training samples.
If you want to change some parameters, check the other options with ```python generate_datasets.py -h```.
```
python generate_dataset.py --save_dir data/synthetic_data --type all --num_samples 1280000 10000 10000
```

### 2. Training
We train the RL model on the synthetic datasets. Check the other options with ```python train.py -h```.
```
python train.py --dataset_path data/synthetic_data/train_dataset.pkl --checkpoint_dir checkpoints/demo_model --batch_size 256 --vehicle_speed 41 --wait_time 0.5 --time_horizon 12 --gpu 0
```

### 3. Validation
We determine the best epoch evaluating the RL-model with greedy decoding on the validation split. Check the other options with ```python valid.py -h```.
```
python valid.py --model_dir checkpoints/demo_model --dataset_path data/synthetic_data/valid_dataset.pkl --gpu 0
```

### 4. Evaluation
The option ```--model_dir <check_point_dir>``` automatically selects the weights at the best epoch. You can also select a specific epoch by ```--model_path model_epoch<epoch>.pth```. Specify only one of the two. If you want to output the visualization, add the option ```--visualize_routes```. Check the other options with ```python eval.py -h```.
```
python eval.py --model_dir checkpoints/demo_model --dataset_path data/synthetic_data/eval_dataset.pkl --vehicle_speed 41 --wait_time 0.5 --time_horizon 12 --gpu 0
```

## Reproducibility
Regarding the synthetic datasets, you can reproduce our experimental results in [reproduce_results.ipynb](./reproduce_results.ipynb).  
Please take a glance at the content via nbviwer: [nbviwer-evrp-eps](https://nbviewer.org/github/ntt-dkiku/evrp-eps/blob/main/reproduce_results.ipynb).

## Licence
Our code is licenced by NTT. Basically, the use of our code is limitted to research purposes. See [LICENSE](./LICENSE) for more details.

## Citation
Preparing...