# Electric Vehicle Routing for Emergency Power Supply: Towards Telecom Base Station Relief

<p align="center">
  <a href="https://ntt-dkiku.github.io/rl-evrpeps/" target="_blank"><img src="https://img.shields.io/badge/Project-page-blue"></a>
  <a href="https://arxiv.org/abs/2404.02448" target="_blank"><img src="https://img.shields.io/badge/arXiv-abs-red"></a>
  <a href="https://www.aamas2024-conference.auckland.ac.nz/" target="_blank"><img src="https://img.shields.io/badge/AAMAS-2024-green"></a>
</p>

This repo is the official implementation of Electric Vehicle Routing for Emergency Power Supply with Deep Reinforcement Learning (AAMAS 2024, extended abstract) and [Electric Vehicle Routing for Emergency Power Supply: Towards Telecom Base Station Relief](https://arxiv.org/abs/2404.02448) (arXiv preprint).

<div><video autoplay loop controls src="https://github.com/ntt-dkiku/evrp-eps/assets/154794155/818b81f7-8e19-40ac-9934-70bc30f13e53"></video></div>

## 📦 Python Environment
We recommend using Docker to construct the python environment. You can use the [Dockerfile](./Dockerfile) in this repository. 
```
docker build -t evrp-eps/evrp-eps:1.0 .
``` 
You can run code interactively with the following command (<> indicates a placeholder, which you should replace according to your settings.").
```
docker run -it --rm -v </path/to/clone/repo>:/workspace/app --name evrp-eps -p <host_port>:<container_port> --gpus all evrp-eps/evrp-eps:1.0 bash
```

## 🔧 Usage
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

## 🧪 Reproducibility
Regarding the synthetic datasets, you can reproduce our experimental results in [reproduce_results.ipynb](./reproduce_results.ipynb).  
Please take a glance at the content via nbviwer: [nbviwer-evrp-eps](https://nbviewer.org/github/ntt-dkiku/evrp-eps/blob/main/reproduce_results.ipynb).
You can open the Jupyter Notebook server with the following command inside the container, then access it from your browser on localhost.
```
jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port <container_port>
```

## 🐞 Bug reports and questions
If you encounter a bug or have any questions, please post issues in this repo.

## 📄 Licence
Our code is licenced by NTT. Basically, the use of our code is limitted to research purposes. See [LICENSE](./LICENSE) for more details.

## 🤝 Citation
If you find this work useful, please cite our paper as follows:
```
@misc{kikuta2024electric,
      title={Electric Vehicle Routing Problem for Emergency Power Supply: Towards Telecom Base Station Relief}, 
      author={Daisuke Kikuta and Hiroki Ikeuchi and Kengo Tajiri and Yuta Toyama and Yuusuke Nakano},
      year={2024},
      eprint={2404.02448},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```
