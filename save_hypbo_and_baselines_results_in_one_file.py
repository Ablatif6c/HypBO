import os

import pandas as pd

if __name__ == "__main__":
    # Define the directory where the data is stored
    data_dir = os.path.join(
        "data",
        "hypbo_and_baselines"
    )

    # Define the list of synthetic function methods
    synthetic_func_methods = [
        "hypbo",
        "lamcts",
        "lamcts+",
        "pibo",
        "random_search",
        "turbo"
    ]

    n = 105  # Number of iterations to keep
    # Create an empty DataFrame to store the data
    data = pd.DataFrame(
        columns=["func_name", "method", "scenario", "value", "iteration"])

    # Iterate over each method
    for method in synthetic_func_methods:
        # Iterate over each function name in the method directory
        for func_name in os.listdir(os.path.join(data_dir, method)):
            if func_name == "HER":
                continue
            # Read the average file
            avg_file = os.path.join(
                data_dir, method, func_name, f"{func_name}_average.csv")
            avg_df = pd.read_csv(avg_file)
            # keep the first 105 iterations
            avg_df = avg_df.iloc[:n]

            # Get the scenario data
            for scenario in avg_df.columns[1:]:
                values = avg_df[scenario].values
                if len(values) > 0:
                    # Create a DataFrame with the data to add
                    data_to_add = pd.DataFrame(
                        {
                            "func_name": [func_name] * len(values),
                            "method": [method] * len(values),
                            "scenario": [scenario] * len(values),
                            "value": values,
                            "iteration": range(1, len(values) + 1)
                        }
                    )
                    # Concatenate the data to the main DataFrame
                    data = pd.concat([data, data_to_add])

    # Rename the columns to match the names in the paper.
    if "Poor_Weak" in data["scenario"].values and "Weak_Poor" in data["scenario"].values:
        data["scenario"] = data["scenario"].replace("Poor_Weak", "Weak_Poor")
    if "Poor_Good" in data["scenario"].values and "Good_Poor" in data["scenario"].values:
        data["scenario"] = data["scenario"].replace("Poor_Good", "Good_Poor")
    if "Weak_Good" in data["scenario"].values and "Good_Weak" in data["scenario"].values:
        data["scenario"] = data["scenario"].replace("Weak_Good", "Good_Weak")

    # Save the data to a CSV file
    data.to_csv(os.path.join(data_dir, "hypbo_and_baselines.csv"))
