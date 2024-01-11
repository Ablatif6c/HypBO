import os
from multiprocessing import Pool, Process
from typing import List, Tuple

import pandas as pd
from test_hypotheses import get_function, get_scenario_name, get_scenarios


def process_scenario(
    path: str,
    scenario_name: str,
    func_name: str,
    objective: float,
    regret: bool = True,
):
    seeds = {}
    # Get the dataset files
    scenario_path = os.path.join(
        path,
        scenario_name,
    )
    if not os.path.exists(scenario_path):
        return

    for file in os.listdir(scenario_path):
        print(f"\t\t...file: {file}")
        if file.startswith(f"dataset_{func_name}_{scenario_name}"):
            # Fetch the seed integer in the filename
            seed = int(next(filter(str.isdigit, file.split("_"))))

            # Create a DataFrame to save all data from
            # all scenarios of that seed
            if seed not in seeds.keys():
                seeds[seed] = pd.DataFrame()

            # Get the Target column
            target = pd.read_csv(
                os.path.join(
                    scenario_path,
                    file,
                )
            )["Target"]
            best_so_far = pd.DataFrame(columns=[scenario_name])

            # Calculate the best so far
            for i in target.index:
                if regret:
                    best_so_far.loc[i] = objective - max(target[: i + 1])
                else:
                    best_so_far.loc[i] = max(target[: i + 1])

            # Save the best so far column
            seeds[seed][scenario_name] = best_so_far[scenario_name]
    return seeds


def save_all_best_so_far(
    func_name: str,
    scenario_names: List[str],
    dir_name: str = "data",
    regret=False,
):
    print(
        f"{dir_name}: generating the best so far datasets for {func_name}...")
    seeds = {}
    path = os.path.join(
        dir_name,
        func_name,
    )

    objective = 0
    if regret:
        if func_name == "HER":
            objective = 28
        else:
            func = get_function(name=func_name.split("_")[0])
            objective = func.problem.yopt
    # For each scenario
    with Pool(processes=len(scenario_names)) as pool:
        args = [
            (path, scenario_name, func_name, objective, regret)
            for scenario_name in scenario_names]
        results = pool.starmap(
            process_scenario,
            args,
        )
        for res in results:
            for seed, series in res.items():
                if seed not in seeds.keys():
                    seeds[seed] = pd.DataFrame()
                scenario_name = series.columns[0]
                seeds[seed][scenario_name] = series

    for seed, df in seeds.items():
        file_path = os.path.join(path, f"all_{func_name}_bsf_{seed}.csv")
        df.to_csv(file_path)
        print(f"Data generated and saved in {file_path}")


def save_avg_and_std_best_so_far(
    func_name: str,
    scenario_names: List[str],
    dir_name: str = "data",
):
    print(f"Generating the average and std datasets for {func_name}...")
    path = os.path.join(
        dir_name,
        func_name,
    )
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
    method_dir: str,
    method: str = "HypBO",
    regret: bool = True,
):
    scenarios_list = get_scenarios(
        func_name=func_name,
        dim=dim,
    )
    scenario_names = []
    if method == "HypBO":
        scenario_names = [get_scenario_name(
            scenarios) for scenarios in scenarios_list]
    else:
        scenario_names = ["No Hypothesis"]

    if "HER" not in func_name:
        func_name = f"{func_name}_d{dim}"
    save_all_best_so_far(
        func_name=func_name,
        dir_name=method_dir,
        scenario_names=scenario_names,
        regret=regret,
    )
    save_avg_and_std_best_so_far(
        func_name=func_name,
        dir_name=method_dir,
        scenario_names=scenario_names,
    )


def base_multiprocess_function_data(
    test_functions: List[Tuple[str, int]],
    method_dir: str,
    method: str = "HypBO",
    regret: bool = True,
):
    processes = []
    for func_name, dim in test_functions:
        process = Process(
            target=process_function_data,
            args=(func_name, dim, method_dir, method, regret),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def all_functions(test_functions: List[Tuple[str, int]], methods: List[str]):
    for method in methods:
        method_dir = os.path.join(
            data_dir,
            method,
        )
        base_multiprocess_function_data(
            test_functions=test_functions, method=method, method_dir=method_dir
        )


def ablation_studies(test_functions: List[Tuple[str, int]], ablation_dir: str):
    ablation_dirs = [dir for dir in os.listdir(
        ablation_dir) if os.path.isdir(os.path.join(ablation_dir, dir))]

    for dir_name in ablation_dirs:
        dir_path = os.path.join(ablation_dir, dir_name)
        base_multiprocess_function_data(test_functions, method_dir=dir_path)


if __name__ == "__main__":
    data_dir = os.path.join("data")
    synthetic_func_methods = ["HypBO", "LA-MCTS", "Random Search"]

    her_func_methods = ["HypBO", "DBO", "Random Search"]
    test_her_functions = [("HER", 10)]

    test_synthetic_functions = [
        ("Branin", 2),
        ("Sphere", 2),
        ("Ackley", 5),
        ("Ackley", 9),
        ("Rosenbrock", 14),
        ("Levy", 20),
    ]

    # all_functions(
    #     test_functions=test_synthetic_functions,
    #     methods=synthetic_func_methods,
    # )
    # all_functions(
    #     test_functions=test_her_functions,
    #     methods=her_func_methods,
    # )

    ablation_dir = os.path.join(data_dir, "Ablation Studies")
    ablation_studies(test_functions=test_synthetic_functions,
                     ablation_dir=ablation_dir)
