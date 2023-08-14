import os

import pandas as pd
from scipy.stats import wilcoxon

data_folder = os.path.join("data")
test_functions = [
    "Branin_d2",
    "Sphere_d2",
    "Ackley_d5",
    "Ackley_d9",
    "Rosenbrock_d14",
    "Levy_d20",
]

synthetic_baselines = [
    ("HypBO", "LA-MCTS"),
    ("HypBO", "Random Search"),
]

synthetic_func_hypotheses = [
    "Poor",
    "Weak",
    "Good",
    "Poor_Good",
    "Weak_Good",
    "Poor_Weak",
]

her_hypotheses = [
    "Perfect Hindsight",
    "What they knew",
    "Bizarro World",
    "Mixed Starting Hypotheses",
]
her_baselines = [
    ("HypBO", "DBO"),
    ("HypBO", "Random Search"),
]


def wilcoxon_test(hypotheses, test_functions, baselines, save_path_prefix=""):
    n = 105
    columns = ["Baseline", "Function"] + hypotheses
    results_statistics = pd.DataFrame(columns=columns)
    results_pvalues = pd.DataFrame(columns=columns)
    for hypbo, baseline in baselines:
        for function in test_functions:
            file_1 = os.path.join(
                data_folder,
                hypbo,
                function,
                f"{function}_average.csv",
            )
            df_hyp = pd.read_csv(file_1).head(n=n)

            file_2 = os.path.join(
                data_folder,
                baseline,
                function,
                f"{function}_average.csv",
            )
            df_baseline = pd.read_csv(file_2).head(n=n)

            row_statistics = [baseline, function]
            row_pvalues = [baseline, function]
            for hypothesis in hypotheses:
                hyp_data = df_hyp[hypothesis].to_numpy()
                baseline_data = df_baseline["No Hypothesis"].to_numpy()

                res = wilcoxon(
                    baseline_data,
                    hyp_data,
                    alternative="less",
                    )
                row_statistics.append(res.statistic)
                row_pvalues.append(res.pvalue)
            results_statistics.loc[len(results_statistics)] = row_statistics
            results_pvalues.loc[len(results_pvalues)] = row_pvalues

    # Save Wilcoxon test results
    results_statistics.to_csv(
        os.path.join(
            data_folder,
            f"{save_path_prefix}_wilcoxon_statistics.csv"),
        )
    results_pvalues.to_csv(
        os.path.join(
            data_folder,
            f"{save_path_prefix}_wilcoxon_pvalues.csv"),
        )


wilcoxon_test(
    hypotheses=synthetic_func_hypotheses,
    test_functions=test_functions,
    baselines=synthetic_baselines,
    save_path_prefix="synthetic_func")

wilcoxon_test(
    hypotheses=her_hypotheses,
    test_functions=["HER"],
    baselines=her_baselines,
    save_path_prefix="HER")
