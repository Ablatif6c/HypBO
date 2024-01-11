"""
This script is used to generate and save the best so far datasets for a given
function and scenario.
"""

import os
from multiprocessing import Pool, Process

import pandas as pd

from utils import get_function


def process_scenario(
    path: str,
    scenario_name: str,
    func_name: str,
    objective: float,
    regret: bool = True,
) -> dict:
    """
    Process a scenario and calculate the best so far values.

    Args:
        path (str): The path to the scenario files.
        scenario_name (str): The name of the scenario.
        func_name (str): The name of the function.
        objective (float): The objective value.
        regret (bool, optional): Whether to calculate regret. Defaults to True.

    Returns:
        dict: A dictionary containing the best so far values for each seed.
    """
    seeds = {}
    # Get the dataset files
    scenario_path = os.path.join(path, scenario_name)
    if not os.path.exists(scenario_path):
        return

    for file in os.listdir(scenario_path):
        print(f"\t\t...file: {file}")
        if file.startswith(f"{func_name.split('_')[0]}_{scenario_name}"):
            # Fetch the seed integer in the filename
            seed = int(next(filter(str.isdigit, file.split("_"))))

            # Create a DataFrame to save all data from
            # all scenarios of that seed
            if seed not in seeds.keys():
                seeds[seed] = pd.DataFrame()

            # Get the Target column
            target = pd.read_csv(os.path.join(scenario_path, file))["Value"]
            best_so_far = pd.DataFrame(columns=[scenario_name])

            # Calculate the best so far
            for i in target.index:
                if regret:
                    best_so_far.loc[i] = min(target[: i + 1]) - objective
                else:
                    best_so_far.loc[i] = min(target[: i + 1])

            # Save the best so far column
            seeds[seed][scenario_name] = best_so_far[scenario_name]
    return seeds


def save_all_best_so_far(
    func_name: str,
    dir_name: str = "data",
    regret: bool = False,
) -> None:
    """
    Save the best so far datasets for all scenarios of a given function.

    Args:
        func_name (str): The name of the function.
        dir_name (str, optional): The directory name. Defaults to "data".
        regret (bool, optional): Whether to calculate regret.
            Defaults to False.
    """
    print(
        f"{dir_name}: generating the best so far datasets for {func_name}..."
    )
    seeds = {}
    path = os.path.join(dir_name, func_name)
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory")

    scenario_names = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder))
    ]
    objective = 0
    if regret:
        func = get_function(name=func_name.split("_")[0])
        objective = func.problem.yopt

    # For each scenario
    with Pool(processes=len(scenario_names)) as pool:
        args = [
            (path, scenario_name, func_name, objective, regret)
            for scenario_name in scenario_names
        ]
        results = pool.starmap(process_scenario, args)
        for res in results:
            for seed, series in res.items():
                if seed not in seeds.keys():
                    seeds[seed] = pd.DataFrame()
                scenario_name = series.columns[0]
                seeds[seed][scenario_name] = series[scenario_name]

    for seed, df in seeds.items():
        file_path = os.path.join(path, f"all_{func_name}_bsf_{seed}.csv")
        df.to_csv(file_path)
        print(f"Data generated and saved in {file_path}")


def save_avg_and_std_best_so_far(
    func_name: str,
    dir_name: str = "data",
) -> None:
    """
    Generate and save the average and standard deviation datasets for a given
    function.

    Args:
        func_name (str): The name of the function.
        dir_name (str, optional): The directory name where the data will be
            saved. Defaults to "data".

    Raises:
        ValueError: If the specified directory does not exist.

    Returns:
        None
    """
    print(f"Generating the average and std datasets for {func_name}...")
    path = os.path.join(
        dir_name,
        func_name,
    )
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory")

    scenario_names = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]
    df_avg = pd.DataFrame(columns=scenario_names)
    df_std = pd.DataFrame(columns=scenario_names)

    for scenario_name in scenario_names:
        # Get the bsf files
        df_scenario = pd.DataFrame()
        for file in os.listdir(path):
            if file.startswith(f"all_{func_name}_bsf_"):
                seed = file.split(".")[0]
                seed = seed.split("_")[-1]
                df = pd.read_csv(
                    os.path.join(
                        path,
                        file,
                    )
                )
                if scenario_name in df.columns:
                    target = df[scenario_name]
                    df_scenario[f"Target_{seed}"] = target
        df_avg[scenario_name] = df_scenario.mean(axis=1)
        df_std[scenario_name] = df_scenario.std(axis=1)

    average_file_path = os.path.join(path, f"{func_name}_average.csv")
    df_avg.to_csv(average_file_path)
    print(f"Data generated and saved in {average_file_path}")

    std_file_path = os.path.join(path, f"{func_name}_std.csv")
    df_std.to_csv(std_file_path)
    print(f"Data generated and saved in {std_file_path}")


def process_function_data(
    func_name: str,
    dim: int,
    data_dir: str,
    regret: bool = True,
):
    """
    Process function data by saving the best-so-far results.

    Args:
        func_name (str): The name of the function.
        dim (int): The dimension of the function.
        data_dir (str): The directory to save the data.
        regret (bool, optional): Flag indicating whether to compute regret.
            Defaults to True.
    """
    func_name = f"{func_name}_d{dim}"
    save_all_best_so_far(
        func_name=func_name,
        dir_name=data_dir,
        regret=regret,
    )
    save_avg_and_std_best_so_far(
        func_name=func_name,
        dir_name=data_dir,
    )


if __name__ == "__main__":
    data_dir = os.path.join("outputs")
    test_synthetic_functions = [
        ("Branin", 2),
        ("Sphere", 2),
        ("Ackley", 5),
        ("Ackley", 9),
        ("Rosenbrock", 14),
        ("Levy", 20)
    ]
    regret = True,
    processes = []
    for func_name, dim in test_synthetic_functions:
        process = Process(
            target=process_function_data,
            args=(func_name, dim, data_dir, regret),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
