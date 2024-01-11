"""
This script generates and saves regret data and cumulative regret data for a
given list of function names and scenario names.

The script takes a list of function names, a list of scenario names, a
directory name to save the data, and the number of data points to consider as
input. It generates regret data and cumulative regret data for each function
and saves them as CSV files.

Example usage:
    dir_name = "outputs"
    func_names = [
        "Branin_d2",
        "Sphere_d2",
        "Ackley_d5",
        "Ackley_d9",
        "Rosenbrock_d14",
        "Levy_d20",
    ]

    scenarios = get_scenarios(
        func_name=func_names[0].split('_')[0],
        dim=int(func_names[0].split('_')[1][1:]),
    )
    scenario_names = [
        scenario.name for scenario in scenarios
    ]

    save_regrets(
        func_names=func_names,
        dir_name=dir_name,
        scenario_names=scenario_names,
    )
"""

import os
from typing import List

import pandas as pd

from utils import get_scenario_name, get_scenarios


def save_regrets(
        func_names: List[str],
        scenario_names: List[str],
        dir_name: str = "data",
        n: int = 105,
):
    """
    Generate and save regret data and cumulative regret data for a given list
    of function names and scenario names.

    Args:
        func_names (List[str]): List of function names.
        scenario_names (List[str]): List of scenario names.
        dir_name (str, optional): Directory name to save the data.
            Defaults to "data".
        n (int, optional): Number of data points to consider.
            Defaults to 105.
    """
    print(f"Generating the regret table for {func_names}...")

    columns = ["Function"] + scenario_names  # Column names
    df_regret = pd.DataFrame(columns=columns)   # Regret dataframe
    # Cumulative regret dataframe
    df_cum_regret = pd.DataFrame(columns=columns)

    for func_name in func_names:
        # Get avg regret
        path = os.path.join(
            dir_name,
            func_name,
            f"{func_name}_average.csv",
        )
        df_func = pd.read_csv(path).head(n=n)   # Get first n rows
        df_func = df_func.drop(df_func.columns[0], axis=1)  # Drop index column
        df_func = df_func.reindex(columns=scenario_names)   # Reorder columns
        row_avg = df_func.iloc[-1]  # Get last row
        row_cum_avg = df_func.sum(axis=0)   # Get cumulative regret

        # Get std regret
        path = os.path.join(
            dir_name,
            func_name,
            f"{func_name}_std.csv",
        )
        df_func = pd.read_csv(path).head(n=n)
        df_func = df_func.drop(df_func.columns[0], axis=1)
        df_func = df_func.reindex(columns=scenario_names)
        row_std = df_func.iloc[-1] / 2
        row_cum_std = df_func.sum(axis=0) / 2

        if "Rosenbrock" in func_name:
            row_avg /= 10000
            row_std /= 10000
            row_cum_avg /= 10000
            row_cum_std /= 10000

        # Regret
        row = [func_name]
        for i in range(len(row_avg)):
            _avg = row_avg[i]
            if _avg < 0.01:
                # Convert to scientific notation if less than 0.01
                _avg = f"{_avg:.2E}"
            else:
                _avg = f"{row_avg[i]:.02f}"  # Format with 2 decimal places

            _std = row_std[i]
            if _std < 0.01:
                # Convert to scientific notation if less than 0.01
                _std = f"{row_std[i]:.2E}"
            else:
                _std = f"{row_std[i]:.02f}"  # Format with 2 decimal places

            # Append formatted average and standard deviation to the row
            row.append(f"{_avg}+-{_std}")
        row = pd.DataFrame(data=[row], columns=columns)
        df_regret = df_regret.append(row)

        # Cumulative regret
        row_cum = [func_name]
        for i in range(len(row_cum_avg)):
            row_cum.append(f"{row_cum_avg[i]:.02f}+-{row_cum_std[i] / 2:.02f}")
        row_cum = pd.DataFrame(data=[row_cum], columns=columns)
        df_cum_regret = df_cum_regret.append(row_cum)

    regret_file_path = os.path.join(
        dir_name,
        f"regret_{'-'.join(func_names)}.csv",
    )
    df_regret.to_csv(regret_file_path)
    print(f"Regret data generated and saved in {regret_file_path}")

    cum_regret_file_path = os.path.join(
        dir_name,
        f"cum_regret_{'-'.join(func_names)}.csv",
    )
    df_cum_regret.to_csv(cum_regret_file_path)
    print(
        f"Cumulative regret data generated and saved in {cum_regret_file_path}"
    )


if __name__ == "__main__":
    dir_name = "outputs"
    func_names = [
        "Branin_d2",
        "Sphere_d2",
        "Ackley_d5",
        "Ackley_d9",
        "Rosenbrock_d14",
        "Levy_d20",
    ]

    scenarios = get_scenarios(
        func_name=func_names[0].split('_')[0],
        dim=int(func_names[0].split('_')[1][1:]),
    )
    scenario_names = [
        get_scenario_name(scenario) for scenario in scenarios
    ]

    save_regrets(
        func_names=func_names,
        dir_name=dir_name,
        scenario_names=scenario_names,
    )
