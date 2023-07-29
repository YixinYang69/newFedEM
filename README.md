#  pFedDef: Defending Grey-Box Attacks for Personalized Federated Learning

This repository is the official implementation of the external evasion attack testbed ([pFedDef](https://arxiv.org/abs/2209.08412)). 

## Summary: 
Personalized federated learning allows for clients in a distributed system to train a neural network tailored to their unique local data while leveraging information from other clients. However, clients' models are vulnerable to attacks during both the training and testing phases. In this paper, we address the issue of adversarial clients crafting evasion attacks at training time. In the context of federated learning, malicious participants may inject poisoned data or model weights into the local training process, aiming to undermine the model's robustness and compromise its accuracy during inference. To counter such threats, Federated Adversarial Training (FAT) demonstrated promising results in bolstering model performance and reducing the susceptibility to manipulation.

The code in this repository has been written to implement evasion attacks at training phase and perform system analysis regarding these attacks.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

All experiments require NVIDIA GPU and CUDA library.
All experiments of this paper are run on the Amazon EC2 instance G4DN.XLARGE.

## Data Sets

The CIFAR-10 and CIFAR-100 data sets are readily available to set up with the existing infrastructure of FedEM. The CelebA data set can be found and downloaded at [LEAF](https://leaf.cmu.edu/), while the MovieLens data set is found at [MovieLens](https://grouplens.org/datasets/movielens/).


## Original Repository

The code from this repository has been heavily adapted from: [Federated Multi-Task Learning under a Mixture of Distributions](https://arxiv.org/abs/2108.10252). The code is found at https://github.com/omarfoq/FedEM.

The following components of their work has been used as a basis for our work.

- data folder with the data set downloading and data splitting
- learner folder with learner and learners_ensemble classes
- 'client.py', 'aggregator.py' classes 


## Our Contribution

We have added the following unique files for experiments:

- Transfer_attacks folder and contents
    - Transfer attacks between one client to another in a federated learning setting
    - Boundary transferer used to measure inter-boundary distance as from: [The Space of Transferable Adversarial Examples](https://arxiv.org/abs/1704.03453)
    - Capability of ensemble attack and alternative transferbility metrics
- 'solve_proportions()' function from 'Transfer_attacks/TA_utils.py' solves the robustness propagation problem given limited resources.
    
The following aspects of the FedEM code has been altered:

- 'client.py' now has adversarial training mechanisms, more detailed implementations of local tuning, and label swapping sybil attack mechanism


## Training

The model weights can be trained given varying methods of adversarial training and aggregation method by running the .py files in the 'run_experiments_collection' folder. Move the .py file to the root directory and follow the instructions to run each experiment.

The scripts have been written in such a way that an individual script can be run for consecutive training of related neural networks. All experiments require NVIDIA GPU and CUDA library, and has been run on AWS EC2 g4dn.xlarge instance.

## Evaluation

## Citation

If you use our code or wish to refer to our results, please use the following BibTex entry:

```
```

## License 

Copyright 2022, Taejin Kim

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

FedEM learning framework copied and adapted from "Federated Multi-Task Learning under a Mixture of Distributions" (https://github.com/omarfoq/FedEM) (Copyright 2021, omarfoq)

Managed by Yixin Yang

## Contact
Taejin Kim
Email: Taejin.Kim@sv.cmu.edu
Yixin Yang
Email: yixinyan@andrew.cmu.edu / yixinyangella@gmail.com
