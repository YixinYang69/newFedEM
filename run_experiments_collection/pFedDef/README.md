 # pFedDef
 
## Introduction

This script runs a pFedDef training on the FedEM model.

## Instructions

### Base usage

For basic usage, change the following arguments in the input group of the run_exp file:
- ```exp_names```: names of experiment when saving weights
- ```args_.method```: training method to use (e.g. 'FedAvg', 'FedAvg_adv', 'FedEM', 'FedEM_adv', 'local', 'local_adv')
- ```args_.num_learners```: number of hypotheses assumed in system (e.g. 3 for FedEM and FedEM_adv, 1 for FedAvg, FedAvg_adv, local, and local_adv)
- ```adv_mode```: whether or not to perfom adv training
- ```args_.experiment```: type of dataset to run (e.g. cifar10, cifar100, celeba)
- ```args_.n_rounds```: number of rounds to run training for
- ```args_.save_path```: Location to save weights
- ```Q```: ADV dataset update freq
- ```G```: adversarial proportion aimed globally
- ```S```: threshold param for robustness propagation
- ```K```: number of steps when generating adv examples
- ```eps```: magnitude of projection during projected gradient descent
- ```args_.aggregation_op```: choose between None, 'median', 'trimmed_mean', and 'krum'

After changing the arguments, train the model with
```python run_experiment_pFedDef.py```
