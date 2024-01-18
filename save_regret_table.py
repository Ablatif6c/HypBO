import os
from typing import List

import pandas as pd


def save_regrets(
    methods: List[str],
    func_names: List[str],
    dir_name: str = "data",
    n: int = 105,
    std=False,
    regret_file_path: str = "regret.csv",
):
    suffix = "std" if std else "average"
    print(f"Computing the {suffix} table.")

    # Add the methods and their scenarios as columns.
    columns = ["Function"]
    for method in methods:
        # Get the first function path.
        first_func_path = os.path.join(
            dir_name,
            method,
            func_names[0],
            f"{func_names[0]}_{suffix}.csv"
        )
        scenario_names = pd.read_csv(
            first_func_path).columns[1:]    # Get scenarios
        for scenario_name in scenario_names:
            # Add the method and scenario as a column.
            columns.append(f"{method} - {scenario_name}")

    # Initialize the regret and cumulative regret dataframes.
    df_regret = pd.DataFrame(columns=columns)
    df_regret["Function"] = func_names  # Add the function names as rows.
    df_cum_regret = pd.DataFrame(columns=columns)
    df_cum_regret["Function"] = func_names  # Add the function names as rows.

    for method in methods:
        for func_name in func_names:
            # Get {suffix} regret.
            path = os.path.join(
                dir_name,
                method,
                func_name,
                f"{func_name}_{suffix}.csv"
            )
            df_func = pd.read_csv(path).head(n=n)   # Get the first n rows.
            # Drop the first column, which is the iteration number.
            df_func = df_func.drop(df_func.columns[0], axis=1)
            # Get the last row, which corresponds to  the smallest regret for
            # each scenario.
            row = df_func.iloc[-1]
            # Add the {suffix} regret to the dataframe.
            for col in row.index:   # For each scenario.
                df_regret.loc[df_regret['Function'] == func_name,
                              f"{method} - {col}"] = row[col]

            # Get the cumulative regret for each scenario.
            row_cum = df_func.sum(axis=0)
            # Add the cumulative regret to the dataframe.
            for col in row_cum.index:   # For each scenario.
                df_cum_regret.loc[df_cum_regret['Function'] == func_name,
                                  f"{method} - {col}"] = row_cum[col]

    # Save the dataframes.
    if not regret_file_path:
        regret_file_path = os.path.join(
            dir_name,
            f"regret_{suffix}_{'-'.join(methods)}.csv"
        )
    df_regret.to_csv(regret_file_path)
    print(f"{suffix} regret data generated and saved in {regret_file_path}")

    cum_regret_file_path = f"{regret_file_path.replace('.csv', '_cum.csv')}"
    df_cum_regret.to_csv(cum_regret_file_path)
    print(
        f"{suffix} cumulative regret data generated and saved in {cum_regret_file_path}"
    )


if __name__ == "__main__":
    data_dir = os.path.join(
        "data",
        "hypbo_and_baselines"
    )
    func_names = [
        "Branin_d2",
        "Sphere_d2",
        "Ackley_d5",
        "Ackley_d9",
        "Rosenbrock_d14",
        "Levy_d20",
    ]
    methods = [
        "random_search",
        "lamcts",
        "lamcts+",
        "pibo",
        "hypbo",
        "turbo"
    ]
    # Save the average regret table.
    regret_file_path = os.path.join(
        data_dir,
        "regret.csv"
    )
    save_regrets(
        methods=methods,
        func_names=func_names,
        dir_name=data_dir,
        regret_file_path=regret_file_path
    )
    # Save the std regret table.
    save_regrets(
        methods=methods,
        func_names=func_names,
        dir_name=data_dir,
        std=True,
        regret_file_path=regret_file_path.replace(".csv", "_std.csv")
    )
