# Intelligent driving intelligence test for autonomous vehicles with naturalistic and adversarial environment

[Michigan Traffic Lab, University of Michigan](https://traffic.engin.umich.edu/)

## Introduction

### About

In this repository, we provide an implementation of our research project 
"Intelligent driving intelligence test for autonomous vehicles 
with naturalistic and adversarial environment". More details 
can be found in the [paper](https://www.nature.com/articles/s41467-021-21007-8).

### Features

Users can use this repo to evaluate AV safety performance on a three-lane highway environment.

### Code Structure

The structure of the code is as follows:

- `NADE_main.py`: the main file to run vehicle testing simulations and generate raw testing data;
- `NADE_core.py`: the main algorithm of NADE;
- `CAV_agent`: the AV agents used in this study that trained by deep reinforcement learning;
- `highway_env`: the highway environment simulator;
- `global_val.py`: the parameters and settings used by the NADE algorithm;
- `config`: the folder where to put config files;
- `data`: the folder carries the static data (e.g., trained AV agent model) used in the repository;
- `plot.ipynb`: the file to analyze testing results and plot figures.


## Acknowledgment

The simulation platform is developed based on an open-source highway simulator `highway-env`. 
For more details of the highway-env, please refer to [highway-env](https://github.com/eleurent/highway-env).


## Installation

1. Create new conda environment

You are recommended to create a new Conda environment to install the project

```bash
conda create -n NADE python=3.7
conda activate NADE
```

2. Pytorch installation

```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
```

3. Clone this repo
```bash
git clone https://github.com/michigan-traffic-lab/NADE.git
cd NADE
```

4. Install all required packages
- install Pytorch according to [official guidance](https://pytorch.org/) 
- install other dependency
```bash
pip install -r requirements.txt
```

## Data

The data (AV2 agent and NDE behavior models, etc) can be downloaded [here](https://intelligent-driving-intelligence-test-for-av-with-nade.s3.amazonaws.com/NADE-data.zip).
Unzip the file and put them into the `Data` folder, the file structure should look like:
```
Repo-path/Data/
|__ AV2_AGENT
|______ ...
|__ CALCULATE_CHALLENGE
|______ ...
|__ NDD_DATA
|______ ...
|__ PLOT_RESULTS
|______ ...
```

## Usage

### Analyze testing results

As the testing result generation is time-consuming, 
for the convenience of usage, we provided example testing results using AV2 agent. The results
are stored in `Data/PLOT_RESULTS/EXAMPLE_NADE_TESTING_RESULTS` folder. It includes around 
500,000 testing episodes, and they are stored in 50 files (each contains around 10,000 episodes).
Each row is a testing episode result, and the data format is:
`Episode id, 
Test result(1: crash, 3: no crash), weight list (likelihood ratio) of each timestep, 
maximum criticality of each timestep, sampled action naturalistic probability of each timestep, 
sampled action importance function probability list of each timestep`.

Use `plot.ipynb` to analyze results and generate crash rate and relative half-width figures. The
plotted figures will be saved in `result` folder by default.

### Run experiment

If you want to generate new testing result, config your desired settings in `configs.yml` (e.g., 
NADE epsilon parameter etc.,) and run the following:

```bash
python NADE_main.py --experiment-name your-experiment-name --folder-idx 1
```
The testing results will be saved in folder `Log` by default.

Note: The `folder-idx` denote the worker id of the running experiment, you can
run multiple experiments (e.g., `folder-idx` from 1 to 16) simultaneously for
parallel computing.

After generating the testing results, you can use `plot.ipynb` to analyze it.  

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Developers

Xintao Yan (xintaoy@umich.edu)

Haowei Sun (haoweis@umich.edu)

Shuo Feng (fshuo@umich.edu)

## License

## Contact

Henry Liu (henryliu@umich.edu)
