import optuna
from src.utils.feature_utils import load_data
from src.utils.model_utils import CatBoostWrapper
from src.optimization.train import run_cross_validation_loop


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

    train, test = load_data('../../data/processed', file_type='parquet')
    TARGET = 'loan_paid_back'
    y = train[TARGET]
    train = train.drop(TARGET, axis=1)

    model_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 10_000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03),
        'depth': 7,
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 10.0, 15.0),
        'random_seed': 42,
        'task_type': 'GPU',
        'early_stopping_rounds': 400,
        'verbose': 500
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


