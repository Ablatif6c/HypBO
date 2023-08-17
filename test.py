import os

import pandas as pd

folders = [
    "No Hypothesis",
]

methods = [
    "LA-MCTS",
]

function = "Ackley_d9"

for method in methods:
    for folder in folders:
        path = os.path.join(
            "data",
            method,
            function,
            folder
        )
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(
                path,
                file
            )
            df = pd.read_csv(file_path)
            df = df.head(105)
            df.to_csv(file_path)
