import optimizer

results = optimizer.run_study(trial_count=2, direction='maximize')
print(results)