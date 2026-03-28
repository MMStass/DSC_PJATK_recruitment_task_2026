import optuna
import pandas as pd
import features
import model_utils
from features import load_data
from src.model_utils import LightGBMWrapper, CatBoostWrapper
from src.train import run_cross_validation_loop


def run_study(name: str, trial_count: int, direction: str):

    study = optuna.create_study(study_name=name, direction=direction)
    study.optimize(objective, n_trials=trial_count)

    results = {
        "study_name": name,
        "best_params": study.best_params,
        "best_value": study.best_value
    }
    return results



def objective(trial):

    train, test = load_data('../data/processed', file_type='parquet')
    TARGET = 'loan_paid_back'
    y = train[TARGET]
    train = train.drop(TARGET, axis=1)

    model_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 10_000,
        'task_type': 'GPU',
        'random_seed': 42,
        'early_stopping_rounds': 300,
        'verbose': 500,  # Wyciszamy logi, żeby Optuna nie zaśmiecała konsoli


        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.015, log=True),
        'depth': trial.suggest_int('depth', 4, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 5.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 0.5),
    }

    cats = train.select_dtypes(include=['string']).columns.tolist()
    model = CatBoostWrapper(model_params, cats)


    _, _, cv_auc, _ = run_cross_validation_loop(
        X=train,
        test=test,
        y=y,
        model=model,
        n_splits=5,
        target_encoding=False,
        pseudo_target_encoding=False
    )

    return cv_auc


