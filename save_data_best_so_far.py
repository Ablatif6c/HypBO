"""
This script performs ablation studies and saves the results per function per
hypothesis in separate CSV files.
"""

import glob
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils import multiprocess_all_function_data


def ablation_studies_bsf(
    test_functions: List[Tuple[str, int]],
    ablation_dir: str
):
    """
    For each ablation test case directory, generate the best so far data
    on the given test functions.

    Args:
        test_functions (List[Tuple[str, int]]): A list of test functions,
            where each function is represented as a tuple of function
            name (str) and function id (int).
        ablation_dir (str): The path to the ablation directory containing the
        data.

    Returns:
        None
    """
    print("Performing ablation studies...")
    # Get the list of ablation test cases directories.
    ablation_dirs = [dir for dir in os.listdir(
        ablation_dir) if os.path.isdir(os.path.join(ablation_dir, dir))]

    # For each ablation test case directory, generate the best so far data.
    for dir_name in ablation_dirs:
        dir_path = os.path.join(ablation_dir, dir_name)
        multiprocess_all_function_data(test_functions, method_dir=dir_path)


def save_results_per_function_per_hypothesis(ablation_dir: str):
    """
    Save the results per function per hypothesis in separate CSV files.

    Args:
        ablation_dir (str): The directory path where the ablation studies are
        located.

    Returns:
        None
    """
    avg_data = {}
    std_data = {}

    # For each ablation study
    ablation_dirs = [dir for dir in os.listdir(
        ablation_dir) if os.path.isdir(os.path.join(ablation_dir, dir))]
    print(f"Scraping {len(ablation_dirs)} ablation studies:")
    for ablation_dir_name in ablation_dirs:
        print(f"\t{ablation_dir_name}")
        dir_path = os.path.join(ablation_dir, ablation_dir_name)

        # For each test function
        for func_name in os.listdir(dir_path):
            # Get avg file and the scenario data.
            avg_file = os.path.join(
                dir_path, func_name, f"{func_name}_average.csv")
            if os.path.exists(avg_file):
                avg_df = pd.read_csv(avg_file)

                # For each scenario
                for column in avg_df.columns[1:]:   # Remove iteration column
                    if column not in avg_data.keys():
                        avg_data[column] = {}
                    if func_name not in avg_data[column].keys():
                        avg_data[column][func_name] = []

                    # Get the values
                    values = avg_df[column].values
                    if len(values) > 0:
                        avg_data[column][func_name].append(
                            (ablation_dir_name, values))

            # std
            std_file = os.path.join(
                dir_path, func_name, f"{func_name}_std.csv")
            if os.path.exists(std_file):
                std_df = pd.read_csv(std_file)

                # For each scenario
                for column in std_df.columns[1:]:  # Remove iteration column
                    if column not in std_data.keys():
                        std_data[column] = {}
                    if func_name not in std_data[column].keys():
                        std_data[column][func_name] = []
                    # Get the values
                    values = std_df[column].values
                    if len(values) > 0:
                        std_data[column][func_name].append(
                            (ablation_dir_name, values))

    # Create summary files
    print("Saving summary files...")
    for scenario_name in avg_data.keys():
        print(f"\t{scenario_name}")
        for func_name in avg_data[scenario_name].keys():
            # Average
            avg_file_path = os.path.join(
                ablation_dir, f"AS_{scenario_name}_{func_name}_avg.csv")
            columns, data = zip(*avg_data[scenario_name][func_name])
            data = np.array(data).T
            df = pd.DataFrame(columns=columns, data=data)
            df.to_csv(avg_file_path)

            # Std
            std_file_path = os.path.join(
                ablation_dir, f"AS_{scenario_name}_{func_name}_std.csv")
            columns, data = zip(*std_data[scenario_name][func_name])
            data = np.array(data).T
            df = pd.DataFrame(columns=columns, data=data)
            df.to_csv(std_file_path)


def save_all_results_in_one_file(ablation_dir: str):
    """
    Save all ablation study results in a single file.

    Args:
        ablation_dir (str): The directory path where the ablation study
        results are located.

    Returns:
        None
    """
    # Get all avg files.
    csv_av_files = glob.glob(os.path.join(ablation_dir, "*_avg.csv"))

    # Create empty dataframe to save all results in.
    all_results = pd.DataFrame()

    # For each avg file create a dataframe and add it to all_results.
    for avg_file in csv_av_files:
        df = pd.read_csv(avg_file)  # Read avg file.
        # Get function name.
        function_name = os.path.basename(avg_file).split(
            '_')[-3] + "_" + os.path.basename(avg_file).split('_')[-2]
        df['Function'] = function_name  # Add function name to dataframe.
        # Get hypotheses names.
        hypotheses_names = re.search(
            r'AS_(.*?)_'+function_name, os.path.basename(avg_file)).group(1)
        # Add hypotheses names to dataframe.
        df['Hypotheses'] = hypotheses_names
        # Add dataframe to all_results.
        all_results = pd.concat([all_results, df], ignore_index=True)

    # Save all_results in the ablation directory
    # Create file path.
    all_results_file = os.path.join(ablation_dir, "all_results.csv")

    all_results = all_results.reindex(columns=[
        all_results.columns[0],
        'u10_l1',
        'u10_l2',
        'u10_l3',
        'u10_l4',
        'u10_l5',
        'u10_l6',
        'u10_l7',
        'u10_l8',
        'u10_l9',
        'u10_l10',
        'u1_l10',
        'u5_l2',
        'Function',
        'Hypotheses']
    )
    all_results.to_csv(all_results_file, index=False)  # Save all_results.


def all_functions(
    test_functions: List[Tuple[str, int]], methods: List[str], data_dir: str
):
    """
    Process all functions for each method and save the data.

    Args:
        test_functions (List[Tuple[str, int]]): A list of test functions,
            where each function is represented as a tuple of its name and
            an integer.
        methods (List[str]): A list of methods to be processed.
        data_dir (str): The directory where the data will be saved.

    Returns:
        None
    """
    for method in methods:
        method_dir = os.path.join(
            data_dir,
            method,
        )
        multiprocess_all_function_data(
            test_functions=test_functions, method_dir=method_dir
        )


if __name__ == "__main__":
    # Synthetic functions
    test_synthetic_functions = [
        ("Branin", 2),
        ("Sphere", 2),
        ("Ackley", 5),
        ("Ackley", 9),
        ("Rosenbrock", 14),
        ("Levy", 20),
    ]

    # Ablation studies
    do_ablation_study_u_l = True
    if do_ablation_study_u_l:
        ablation_dir = os.path.join("data", "ablation_studies", "u_l")
        # ablation_studies_bsf(test_functions=test_synthetic_functions,
        #                     ablation_dir=ablation_dir)
        # save_results_per_function_per_hypothesis(ablation_dir=ablation_dir)
        save_all_results_in_one_file(ablation_dir=ablation_dir)

    # Benchmarks
    benchmark = False
    if benchmark:
        data_dir = os.path.join("data", "hypbo_and_baselines")
        synthetic_func_methods = ["hypbo"]  # , "lamcts", "random_search"]

        all_functions(
            test_functions=test_synthetic_functions,
            methods=synthetic_func_methods,
            data_dir=data_dir,
        )

        # her_func_methods = ["HypBO", "DBO", "Random Search"]
        # test_her_functions = [("HER", 10)]
        # all_functions(
        #     test_functions=test_her_functions,
        #     methods=her_func_methods,
        #     data_dir=data_dir,
        # )
