# HypBO: Accelerating Black-Box Scientific Experiments using Experts' Hypotheses

## Overview

This repository contains the source code and associated materials for the HypBO algorithm from the **_HypBO: Accelerating Black-Box Scientific Experiments using Experts' Hypotheses_** paper appearing in IJCAI 2024. This codebase allows for the replication of the experiments and findings presented in the paper.

Bayesian optimization (BO) is a popular optimization method for scientific problems. However, the search spaces can be large and daunting. To solve this, we propose Hypothesis Bayesian Optimization (HypBO), which uses expert human-defined hypotheses to generate improved seed samples to feed into BO. This improves the search performance and has been validated on various synthetic functions and a real chemical design task.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

To ensure consistency in software environments, HypBO's environment is a Docker container whose Dockerfile is located in the root folder. Docker is a tool that simplifies application creation, deployment, and running using containers. Unlike Python environments, these containers encapsulate an application’s entire runtime environment, including system libraries and settings. Because Docker containers are not affected by variations in host system configurations, Docker is well-suited for replicating complex applications' environments, which reduces "it works on my machine" problems, a level of consistency that Python virtual environments can't provide on their own.

Moreover, each baseline method used for comparison has a separate environment to avoid package conflicts between baselines. A baseline's environment is a Docker container, as well, whose Dockerfile is saved in the baseline's folder in the code repository.

You are not required to user Docker, but it is recommended. You can still use a Python virtual environment if you don't want to use Docker.

### Installation

After downloading the HypBO repository, the installation steps depends on whether you decide to use Docker or a Python virtual environment to set up the development and testing environment.

#### Using Docker

1. Visit the [Docker website](https://docs.docker.com/get-docker/).
2. Choose the appropriate version of Docker for your operating system (e.g., Docker Desktop for Windows or Docker Desktop for Mac).
3. Follow the instructions provided on the Docker website to download the Docker installer.
4. Once the installer is downloaded, run it and follow the on-screen instructions to install Docker on your machine.
5. After the installation is complete, open Docker and ensure that it is running properly.
6. You may need to configure Docker settings based on your preferences and system requirements.
7. Verify the installation by opening a terminal or command prompt and running the command docker --version. You should see the Docker version information displayed.
8. Open a terminal or command prompt.
9. Navigate to the HypBO directory that contains the Dockerfile. You can do this with the `cd` command. For example, you would type `cd HypBO`.
10. Once you're in the HypBO directory, you can build the Docker image with the `docker build` command by typing `docker build -t hypbo .`.
11. After the image is built, you can create a container from it with the `docker run` command. For example, `docker run -d -p 8080:80 hypbo`. This is what each part of this command command does:

    - `-d`: This option runs the container in detached mode, which means it runs in the background.
    - `-p 8080:80`: This option maps port 8080 on your machine to port 80 on the Docker container. You can change these numbers to whatever ports you want to use.

#### Using a Python virtual environment

1. First, make sure you have Python3.8 installed on your machine. If not, you can download it from the [Python website](https://www.python.org/downloads/).
2. Open a terminal or command prompt.
3. Navigate to the HypBO directory where to create the virtual environment. You can do this with the `cd` command by typing `cd HypBO`.
4. Once you're in the directory, you can create the virtual environment with the `python -m venv` command by running `python3.8 -m venv .venv`.
5. After the virtual environment is created, you can activate it. The command to do this depends on your operating system:
   - On Windows, type `.hypbo\Scripts\activate`.
   - On Unix or MacOS, type `source .hypbo/bin/activate`.
6. Once the `.venv` virtual environment is activated, your terminal or command prompt should show the name of the virtual environment.
7. You can now install the required packages into the `.venv` virtual environment using `pip install -r requirements.txt`.

## Usage

Note that HypBO is a maximizing optimizer. Please make sure you reformulate potential minimization problems.

### Synthetic Functions

The synthetic function for testing optimization algorithms are taken from the [Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/optimization.html) and implemented in the `resources` folder where you will find an exhaustive list of those synthetic functions.

To run the synthetic function experiments, run the file `main_continuous.py`. It runs the HypBO algorithm for the given:

- function `func_name`
- function dimension `dim`
- starting seed `seed_start`. Defaults to 0.
- seed count `seed_count`. Defaults to 2.
- initial sample count `n_init`. Defaults to 5.
- budget `budget`. Defaults to 10.
- batch size `batch`. Defaults to 1.

It also saves the optimization data in `data/hypbo_and_baselines/hypbo/{func_name}_{dim}`.

For example to run the Branin d2 function with 5 initial samples, a budget of 100 iterations and 10 trials, run: `python main_continuous.py --func_name Branin --dim 2 --n_init 5 --budget 100 --seed_count 10`.

### Photocatalytic Hydrogen Production

To run the Photocatalytic Hydrogen Production experiment, run the file `main_discrete.py`. It runs the HypBO algorithm for the given:

- seed `seed`. Defaults to 0.
- initial sample count `n_init`. Defaults to 5.
- budget `budget`. Defaults to 10.
- batch size `batch`. Defaults to 1.

It also saves the optimization data in `data/hypbo_and_baselines/hypbo/HER`.

For example to run the Photocatalytic Hydrogen Production experiment with 10 initial samples and a budget of 100 iterations run: `python main_discrete.py --n_init 5 --budget 100`.

### Baselines methods

Details on how to use the baseline methods can be found in their respective sub-folders in `baselines/`.

## Citation

Please cite us as:

```
@inproceedings{cisse2024hypbo,
  title = {HypBO: Accelerating Black-Box Scientific Experiments using Experts' Hypotheses},
  author = {Cisse, Abdoulatif and Evangelopoulos, Xenophon and Carruthers, Sam and Gusev, Vadimir V and Cooper, Andrew I},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  year      = {2024},
  note      = {Main Track}
}
```

An extended version of the paper which includes the Supplementary Material can be found at [arXiv:2308.11787](https://arxiv.org/abs/2308.117870).

## License

[MIT license](https://opensource.org/license/mit/)
