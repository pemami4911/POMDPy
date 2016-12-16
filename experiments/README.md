# Experiment Notes

## Baselines

```python
{'beta': 0.001,
 'discount': 0.95,
 'env': 'Tiger',
 'epsilon_decay': 0.95,
 'epsilon_decay_step': 20,
 'epsilon_minimum': 0.1,
 'epsilon_start': 0.5,
 'learning_rate': 0.0025,
 'learning_rate_decay': 0.96,
 'learning_rate_decay_step': 10,
 'learning_rate_minimum': 0.0025,
 'max_steps': 200,
 'n_epochs': 1000,
 'planning_horizon': 5,
 'preferred_actions': False,
 'save': False,
 'seed': 1993,
 'solver': 'VI-Baseline',
 'test': 10,
 'use_tf': False}
```

Running classic VI agent with planning horizon of 8...

epochs: 1000
ave undiscounted return/epoch: 4.39679369829 +- 0.136399084841
ave discounted return/epoch: 3.86836842853 +- 0.126652175681
ave time/epoch: 0.000742731332779
wrong door count: 87

Running classic VI agent with planning horizon of 1...

epochs: 1000
ave undiscounted return/epoch: 5.26891697607 +- 0.167387410448
ave discounted return/epoch: 4.95916315759 +- 0.158979475964
ave time/epoch: 7.1573972702e-05
wrong door count: 127

Running random agent...

epochs: 1000
ave undiscounted return/epoch: -6.91605000138 +- 0.332756869777
ave discounted return/epoch: -6.71346215429 +- 0.324032687787
ave time/epoch: 3.98526191711e-05
wrong door count: 516

## Linear Alpha Net

Best results so far: 

```python
{'beta': 0.001,
 'discount': 0.95,
 'env': 'Tiger',
 'epsilon_decay': 0.99,
 'epsilon_decay_step': 75,
 'epsilon_minimum': 0.02,
 'epsilon_start': 0.2,
 'learning_rate': 0.01,
 'learning_rate_decay': 0.996,
 'learning_rate_decay_step': 50,
 'learning_rate_minimum': 0.00025,
 'max_steps': 50,
 'n_epochs': 1000,
 'planning_horizon': 5,
 'preferred_actions': False,
 'save': False,
 'seed': 1993,
 'solver': 'LinearAlphaNet',
 'test': 10,
 'use_tf': True}
```

* mean undiscounted return per epoch was `4.251`
* std dev for undiscounted return was `9.486`
* mean discounted return per epoch was `3.997`
* std dev for discounted return was `9.008`
* wrong door count: 16

### TODO 
* Multiple experiments with different random seeds

### Results
* experiments/results/LinearAlphaNet-best


