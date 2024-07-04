import optuna
from src.train import train
import yaml

def objective(trial):
    config = {
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'l1_lambda': trial.suggest_loguniform('l1_lambda', 1e-5, 1e-2),
        'epochs': 100
    }
    return train(config)

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the best configuration
    best_config = {**yaml.safe_load(open("configs/default_config.yaml")), **trial.params}
    with open("configs/best_config.yaml", "w") as f:
        yaml.dump(best_config, f)