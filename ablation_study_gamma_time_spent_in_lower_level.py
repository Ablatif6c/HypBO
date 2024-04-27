
"""
This script calculates the time spent in the lower level for different
scenarios of the gamma ablation study. The script reads CSV files from
specified directories, processes the data, and saves the results to a CSV file.
"""


import os

import numpy as np
import pandas as pd

gamma_ablation_study_dir = os.path.join(
    "data", "ablation_studies", "gamma")

# Get gamma folders in gamma_ablation_study_dir
gamma_dirs = [f.path for f in os.scandir(
    gamma_ablation_study_dir) if f.is_dir()]

time_spent = {}  # gamma -> function -> scenario -> time spent in lower level

# Iterate over the gamma folders
for gamma_dir in gamma_dirs:
    time_spent[os.path.basename(gamma_dir)] = {}

    # Get the functions folders in gamma folder
    func_folders = [f.path for f in os.scandir(
        gamma_dir) if f.is_dir()]

    # Iterate over the functions folders
    for func_folder in func_folders:
        # Get the scenarios folders in func folder
        scenario_folders = [f.path for f in os.scandir(
            func_folder) if f.is_dir()]

        time_spent[os.path.basename(
            gamma_dir)][os.path.basename(func_folder)] = {}

        # Iterate over the scenarios folders
        for scenario_folder in scenario_folders:
            time_spent[os.path.basename(gamma_dir)][os.path.basename(
                func_folder)][os.path.basename(scenario_folder)] = []
            # Get the files in the folder
            files = [f.path for f in os.scandir(
                scenario_folder) if f.is_file()]

            # Iterate over the files
            for i, file in enumerate(files):
                df = pd.read_csv(file)
                df.to_csv(file, index=False)
                # Delete the first 5 rows corresponding to the warmup
                df = df.iloc[5:]
                # Get the ratio of iterations with global vs total iterations
                count_global_hypothesis = df[df["Hypothesis"]
                                             == "Global"].shape[0]
                total_rows = df.shape[0]

                time_spent[
                    os.path.basename(gamma_dir)][
                        os.path.basename(func_folder)][
                            os.path.basename(scenario_folder)].append(
                    1 - count_global_hypothesis / total_rows)
            time_spent[
                os.path.basename(gamma_dir)][
                    os.path.basename(func_folder)][
                        os.path.basename(scenario_folder)] = np.mean(
                            time_spent[
                                os.path.basename(gamma_dir)][
                                    os.path.basename(func_folder)][
                                        os.path.basename(scenario_folder)]
            )

# Transform time_spent to a DataFrame
transformed = {}
for gamma, functions in time_spent.items():
    transformed[gamma] = {}
    for func, scenarios in functions.items():
        for scenario, value in scenarios.items():
            if scenario not in transformed[gamma]:
                transformed[gamma][scenario] = []
            transformed[gamma][scenario].append(value)

# Calculate mean functions of each scenario
columns = ["gamma"] + list(transformed["0"].keys())
time_spent_df = pd.DataFrame(columns=columns)
time_spent_df["gamma"] = list(transformed.keys())
time_spent_df.set_index("gamma", inplace=True)

for gamma, scenarios in transformed.items():
    for scenario, values in scenarios.items():
        transformed[gamma][scenario] = np.mean(values)
        time_spent_df.loc[gamma, scenario] = transformed[gamma][scenario]

# Reset the index to make gamma a column
time_spent_df.reset_index(inplace=True)

# Save the DataFrame to a CSV file
time_spent_df.to_csv(
    os.path.join(
        "data", "ablation_studies", "gamma", "time_spent.csv"), index=False)
