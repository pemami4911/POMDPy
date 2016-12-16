# Experiment Notes

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
### TODO 
* Multiple experiments with different random seeds

### Results
* experiments/results/LinearAlphaNet-best 


