import os

import numpy as np
import pandas as pd


def summarise(ablation_dir: str):
    avg_data = {}
    std_data = {}

    # For each ablation study
    ablation_dirs = [dir for dir in os.listdir(
        ablation_dir) if os.path.isdir(os.path.join(ablation_dir, dir))]
    print(f"Scraping {len(ablation_dirs)}# ablation studies:")
    for ablation_dir_name in ablation_dirs:
        print(f"\t{ablation_dir_name}")
        dir_path = os.path.join(ablation_dir, ablation_dir_name)

        # For each test function
        for func_name in os.listdir(dir_path):
            # Get avg file and the scenario data.
            avg_file = os.path.join(
                dir_path,
                func_name,
                f"{func_name}_average.csv"
            )
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
                dir_path,
                func_name,
                f"{func_name}_std.csv"
            )
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
                ablation_dir,
                f"AS_{scenario_name}_{func_name}_avg.csv")
            columns, data = zip(*avg_data[scenario_name][func_name])
            data = np.array(data).T
            df = pd.DataFrame(columns=columns, data=data)
            df.to_csv(avg_file_path)

            # Std
            std_file_path = os.path.join(
                ablation_dir,
                f"AS_{scenario_name}_{func_name}_std.csv")
            columns, data = zip(*std_data[scenario_name][func_name])
            data = np.array(data).T
            df = pd.DataFrame(columns=columns, data=data)
            df.to_csv(std_file_path)


if __name__ == "__main__":
    data_dir = os.path.join("data")
    ablation_dir = os.path.join(data_dir, "Ablation Studies")
    summarise(ablation_dir=ablation_dir)
