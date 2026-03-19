import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import os

LOG_FILE = 'experiment_logs.csv'

def log_experiment(experiment_name, cv_score, num_features, notes = ""):
    log_entry = pd.DataFrame([{
        'experiment_name': experiment_name,
        'cv_auc': cv_score,
        "num_features": num_features,
        "notes": notes
    }])

    if not os.path.isfile(LOG_FILE):
        log_entry.to_csv(LOG_FILE, index=False)
    else:
        log_entry.to_csv(LOG_FILE, mode = 'a', header = False, index = False)
    print(f"--> Logged {experiment_name} with AUC: {cv_score:.5f}")

def run_lgb_experiment(train_df, features_list, model_params, experiment_name, target_col='loan_paid_back', n_splits=5):
    print(f"\nRunning Experiment: {experiment_name}")
    print(f"Features count: {len(features_list)}")

    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    oof_predictions = np.zeros(len(train_df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[target_col])):
        X_train = train_df.iloc[train_idx][features_list]
        y_train = train_df.iloc[train_idx][target_col]
        X_val = train_df.iloc[val_idx][features_list]
        y_val = train_df.iloc[val_idx][target_col]

        model = lgb.LGBMClassifier(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )

        oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]

        del X_train, y_train, X_val, y_val, model
        gc.collect()

    cv_score = roc_auc_score(train_df[target_col], oof_predictions)

    oof_df = pd.DataFrame({'id': train_df.index, 'oof_pred': oof_predictions})
    oof_df.to_csv(f"oof_{experiment_name}.csv", index=False)

    log_experiment(experiment_name, cv_score, len(features_list))

    return cv_score, oof_predictions

#Sposób użycia
# lgbm_params = {
#     'objective': 'binary',
#     'metric': 'auc',
#     'learning_rate': 0.05,
#     'num_leaves': 31,
#     'max_depth': 6,
#     'random_state': 42,
#     'n_estimators': 2000,
#     'verbose': -1,
#     'n_jobs': -1
# }
#
# all_generated_features = [c for c in train.columns if c != 'loan_paid_back']
#
# features_to_drop = [
#     'grade_subgrade_d1', 'employment_status', 'loan_purpose', 'TE_grade_subgrade',
#     'loan_amount_mean_by_employment_status', 'marital_status', 'gender',
#     'loan_amount_mean_by_education_level', 'education_level',
#     'annual_income_mean_by_grade_subgrade', 'loan_amount_mean_by_grade_subgrade',
#     'annual_income_mean_by_education_level'
# ]
#
# pruned_features = [f for f in all_generated_features if f not in features_to_drop]
#
# cv_auc_pruned, oof_pruned = run_experiment(
#     train_df=train,
#     features_list=pruned_features,
#     model_params=lgbm_params,
#     experiment_name="lgbm_003_pruned_features"
# )