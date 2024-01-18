import optuna
from hypbo import HypBO
from joblib import Parallel, delayed

def objective(trial):
    # Define the range of gamma values using log scale
    gamma = trial.suggest_float("gamma", 0.01, 100, log=True)

    # Define the types of hypotheses
    hypothesis_type = trial.suggest_categorical("hypothesis_type", ["good", "weak", "poor", "mixture"])

    # Initialize the HypBO object with selected gamma and hypothesis type
    hypbo_params = {
        func=func,
        feature_names=feature_names,
        model=BayesianOptimization,
        model_kwargs=model_kwargs,
        hypotheses=scenarios,
        seed=seed,
        n_processes=n_processes,
        discretization=False,
        'global_failure_limit': 5,
        'local_failure_limit': 2
    }
    hypbo_params = {
        'gamma': gamma,
        'hypothesis_type': hypothesis_type,
        
        }
    hypbo = HypBO(hypbo_params)

    # Run the optimization and return the performance metric
    performance_metric = hypbo.run_optimization()  # Replace with actual method
    return performance_metric

def main():
    study = optuna.create_study(direction='maximize')

    # Run the study using multiple processes
    # Set n_jobs to -1 to use all available CPUs
    study.optimize(
        objective,
        n_trials=1,
        n_jobs=-1 # n_jobs to -1 to use all available CPUs
        )

    print(study.best_params)

if __name__ == "__main__":
    main()
