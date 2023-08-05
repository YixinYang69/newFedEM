# Model Replacement Attack
 
## Introduction

This script replaces the global model of some clients with a malicious model.

## Instructions

### Base usage

For basic usage, change the following arguments in the input group of the run_exp file:
- ```exp_names```: names of experiment when saving weights
- ```exp_method```: training method to use (e.g. 'FedAvg', 'FedAvg_adv', 'FedEM', 'FedEM_adv', 'local', 'local_adv')
- ```exp_num_learners```: number of hypotheses assumed in system (e.g. 3 for FedEM and FedEM_adv, 1 for FedAvg, FedAvg_adv, local, and local_adv)
- ```adv_mode```: whether or not to perfom adv training
- ```args_.experiment```: type of dataset to run (e.g. cifar10, cifar100, celeba)
- ```args_.n_rounds```: number of rounds to run training for
- ```args_.save_path```: Location to save weights
- ```num_clients```: number of  clients (e.g. 40 for cifar 10, 50 for cifar 100)
- ```Q```: ADV dataset update freq
- ```G```: adversarial proportion aimed globally
- ```S```: threshold param for robustness propagation
- ```K```: number of steps when generating adv examples
- ```eps```: magnitude of projection during projected gradient descent
- ```num_classes```: number of classes in the data set we are training with
- ```atk_count```: number of clients performing label swap attack

After changing the arguments, train the model with
```python run_experiment_labelatk.py```
