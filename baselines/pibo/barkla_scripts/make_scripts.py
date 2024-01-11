import os

tests = [
    {
        "func_name": "Rosenbrock",
        "dim": 14,
        "seed_start": 0,
        "seed_count": 50,
        "budget": 100,
        "n_init": 5,
    },
]

# Read template
template_file_path = os.path.join(
    "barkla_scripts",
    "template.sh",
)

template = None
with open(template_file_path, 'r') as f:
    template = f.read()

# Create main file
folder_path = os.path.join(
    "barkla_scripts",
)

_seed_count = 10
for test in tests:
    if test["seed_count"] < _seed_count:
        _seed_count = test["seed_count"]
    for seed_start in range(
            test["seed_start"], test["seed_start"] + test["seed_count"],
            _seed_count):
        content = template
        content = content.replace("$job_prefix$", test["func_name"][0])
        content = content.replace("$n_cores$", "6")
        test["seed_count"] = _seed_count
        test["seed_start"] = seed_start
        for key, value in test.items():
            content = content.replace(f"${key}$", str(value))
        file_name = f"job_barkla_{test['func_name']}_d{test['dim']}_{seed_start}_{_seed_count}.sh"
        path = os.path.join(
            folder_path,
            file_name,
        )
        with open(path, 'w') as f:
            f.write(content)
