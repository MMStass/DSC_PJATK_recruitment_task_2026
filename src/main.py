import json
import optimizer

results = optimizer.run_study(name='test1', trial_count= 20, direction='maximize')

with open(f'../logs/Optuna parameters/{results['name']}.json', 'w', encoding='utf-8') as file:
    json.dump(results, file, ensure_ascii=False, indent=4)