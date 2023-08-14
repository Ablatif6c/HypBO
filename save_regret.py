import os
from typing import List

import pandas as pd

from test_hypotheses import get_scenarios, get_scenarios_name


def save_regrets(
        func_names: List[str],
        scenario_names: List[str],
        dir_name: str = "data",
        n: int = 105,):
    print(f"Generating the regret table for {func_names}...")

    columns = ["Function", "RS"] + scenario_names
    df_regret = pd.DataFrame(columns=columns)
    df_cum_regret = pd.DataFrame(columns=columns)

    for func_name in func_names:
        # Get avg regret
        path = os.path.join(
                    dir_name,
                    func_name,
                    f"{func_name}_average.csv",)
        df_func = pd.read_csv(path).head(n=n)
        df_func = df_func.drop(df_func.columns[0], axis=1)
        row_avg = df_func.iloc[-1]
        row_cum_avg = df_func.sum(axis=0)

        # Get std regret
        path = os.path.join(
                    dir_name,
                    func_name,
                    f"{func_name}_std.csv",)
        df_func = pd.read_csv(path).head(n=n)
        df_func = df_func.drop(df_func.columns[0], axis=1)
        row_std = df_func.iloc[-1]/2
        row_cum_std = df_func.sum(axis=0)/2

        # Get avg regret - Random Search
        path = os.path.join(
                    dir_name,
                    "Random Search",
                    func_name,
                    f"{func_name}_average.csv",)
        df_func = pd.read_csv(path).head(n=n)
        rand_avg = df_func.loc[:, "No Hypothesis"].iloc[-1]
        rand_cum_avg = df_func.loc[:, "No Hypothesis"].sum(axis=0)

        # Get std regret
        path = os.path.join(
                    dir_name,
                    "Random Search",
                    func_name,
                    f"{func_name}_std.csv",)
        df_func = pd.read_csv(path).head(n=n)
        rand_std = df_func.loc[:, "No Hypothesis"].iloc[-1]/2
        rand_cum_std = df_func.loc[:, "No Hypothesis"].sum(axis=0)/2

        if "Rosenbrock" in func_name:
            row_avg /= 10000
            row_std /= 10000
            row_cum_avg /= 10000
            row_cum_std /= 10000
            rand_std /= 10000
            rand_avg /= 10000

        # Regret
        row = [func_name, f"{rand_avg:.02f}+-{rand_std:.02f}"]
        for i in range(len(row_avg)):
            _avg = row_avg[i]
            if _avg < 0.01:
                _avg = f"{_avg:.2E}"
            else:
                _avg = f"{row_avg[i]:.02f}"
            _avg = f"{row_avg[i]:.02f}"

            _std = row_std[i]
            if _std < 0.01:
                _std = f"{row_std[i]:.2E}"
            else:
                _std = f"{row_std[i]:.02f}"
            row.append(f"{_avg}+-{_std}")
        row = pd.DataFrame(data=[row], columns=columns)
        df_regret = df_regret.append(row)

        # Cumulative regret
        row_cum = [func_name, f"{rand_cum_avg:.02f}+-{rand_cum_std/2:.02f}"]
        for i in range(len(row_cum_avg)):
            row_cum.append(f"{row_cum_avg[i]:.02f}+-{row_cum_std[i]/2:.02f}")
        row_cum = pd.DataFrame(data=[row_cum], columns=columns)
        df_cum_regret = df_cum_regret.append(row_cum)

    regret_file_path = os.path.join(
                            dir_name,
                            f"regret_{'-'.join(func_names)}.csv",)
    df_regret.to_csv(regret_file_path)
    print(f"Regret data generated and saved in {regret_file_path}")

    cum_regret_file_path = os.path.join(
                            dir_name,
                            f"cum_regret_{'-'.join(func_names)}.csv",)
    df_cum_regret.to_csv(cum_regret_file_path)
    print(
        f"Cumulative regret data generated and saved in {cum_regret_file_path}"
        )


if __name__ == "__main__":

    dir_name = "data"
    func_names = [
        "Branin_d2",
        "Sphere_d2",
        "Ackley_d5",
        "Ackley_d9",
        "Rosenbrock_d14",
        "Levy_d20",
    ]

    scenarios_list = get_scenarios(
            func_name=func_names[0].split('_')[0],
            dim=int(func_names[0].split('_')[1][1:]),
            )
    scenario_names = [
        get_scenarios_name(scenarios) for scenarios in scenarios_list
        ]

    save_regrets(
        func_names=func_names,
        dir_name=dir_name,
        scenario_names=scenario_names,
    )
