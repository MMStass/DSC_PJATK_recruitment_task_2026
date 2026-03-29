import json
import optimizer

results = optimizer.run_study(name='exodia_test', trial_count = 5, direction='maximize')

with open(f'../logs/Optuna parameters/{results['study_name']}.json', 'w', encoding='utf-8') as file:
    json.dump(results, file, ensure_ascii=False, indent=4)