import optuna
import pandas as pd
import features
import model_utils
from features import load_data
from src.model_utils import LightGBMWrapper
from src.train import run_cross_validation_loop


def run_study(name: str,trial_count: int, direction: str):

    study = optuna.create_study(study_name=name, direction=direction)
    study.optimize(objective, n_trials=trial_count)

    results = {
        "study_name": name,
        "best_params": study.best_params,
        "best_trial": study.best_trial,
        "best_value": study.best_value
    }
    return results



def objective(trial):

    train, test = load_data('../data/processed', file_type='parquet')
    TARGET = 'loan_paid_back'
    y = train[TARGET]
    train = train.drop(TARGET, axis=1)


    val = trial.suggest_categorical("use_binned_features", ["True", "False"])

    if val == "False":
        columns_to_drop = [col for col in train.columns.tolist() if col.endswith("_binned")]
        train = train.drop(columns_to_drop, axis=1)


    model_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03),
        "n_estimators": trial.suggest_int("n_estimators", 8000, 10000),
        "max_depth": trial.suggest_int("max_depth", 4, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 0.7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.3),
        "reg_alpha": trial.suggest_float("reg_alpha", 7.5, 12.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 13.5, 16.5),
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    }

    model = LightGBMWrapper(model_params)


    _, _, cv_auc, _ = run_cross_validation_loop(
        X=train,
        test=test,
        y=y,
        model=model,
        n_splits=5,
        target_encoding=True,
        pseudo_target_encoding=True
    )

    return cv_auc


