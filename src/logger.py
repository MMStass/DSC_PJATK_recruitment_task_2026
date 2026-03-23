import os
import json
import pandas as pd
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_dir = '../logs'):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.summary_file = os.path.join(self.log_dir, 'experiments.csv')

    def log_experiment(self, exp_name, model_name, cv_score, params, features_list, importances_df = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{timestamp}_{exp_name}"

        # Save summary to CSV
        summary_data = {
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "model": model_name,
            "cv_score": cv_score,
            "num_features": len(features_list),
            "timestamp": timestamp
        }
        summary_df = pd.DataFrame([summary_data])

        if os.path.exists(self.summary_file):
            summary_df.to_csv(self.summary_file, mode = 'a', header = False, index = False)
        else:
            summary_df.to_csv(self.summary_file, index = False)

        # Save detailed configuration to JSON
        details = {
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "model": model_name,
            "cv_score": cv_score,
            "params": params,
            "features": list(features_list)
        }
        details_path = os.path.join(self.log_dir, f"{exp_id}_details.json")
        with open(details_path, 'w') as f:
            json.dump(details, f, indent  = 4)

        # Save feature importances to CSV (if provided)
        if importances_df is not None and not importances_df.empty:
            importances_path = os.path.join(self.log_dir, f"{exp_id}_importances.csv")
            importances_df.to_csv(importances_path, index=False)

        print(f"\n[Logger] Successfully logged experiment: {exp_id}")
        print(f"[Logger] CV AUC: {cv_score:.5f}")