"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
from utils.utils import *
from utils.constants import *
from utils.args import *
from run_experiment import * 

from torch.utils.tensorboard import SummaryWriter

# Import General Libraries
import os
import argparse
import torch
import copy
import pickle
import random
import numpy as np
import pandas as pd
from models import *

# Import Transfer Attack
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Transferer import *
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import *

import numba 


if __name__ == "__main__":
    
#     exp_names = ['FedAvg_adv_unl', 'FedAvg_adv']
#     unl_mode = [True, False]
#     n_vals = [1,1]
    
    exp_name = '22_09_26_non_participant_sweep_try2/'
    unl_mode = True
    n_vals = 1
    num_unl_clients = [0,5,10,15,20]
    num_clients = 40
    
        
    # Manually set argument parameters
    args_ = Args()
    args_.experiment = "cifar10"
    args_.method = "FedAvg_adv"
    args_.decentralized = False
    args_.sampling_rate = 1.0
    args_.input_dimension = None
    args_.output_dimension = None
    args_.n_learners= 1
    args_.n_rounds = 150 # Reduced number of steps
    args_.bz = 128
    args_.local_steps = 1
    args_.lr_lambda = 0
    args_.lr =0.03
    args_.lr_scheduler = 'multi_step'
    args_.log_freq = 10
    args_.device = 'cuda'
    args_.optimizer = 'sgd'
    args_.mu = 0
    args_.communication_probability = 0.1
    args_.q = 1
    args_.locally_tune_clients = False
    args_.seed = 1234
    args_.verbose = 1
    args_.validation = False
    args_.save_freq = 10
    args_.aggregation_op = None

    # Other Argument Parameters
    Q = 10 # update per round
    G = 0.5 # 0.15 cifar 10, 0.5 cifar 100 
    num_clients = 40 
    S = 0.05 # Threshold
    step_size = 0.01
    K = 10
    eps = 0.1
    prob = 0.8
    Ru = np.ones(num_clients)
    alpha_val_str = '100'
#     num_unlearning_clients = 20
                
    for itt in range(len(num_unl_clients)):
        print("running trial:", itt, "out of", len(num_unl_clients)-1)
        
        args_.save_path = 'weights/unlearning/cifar10/' + exp_name + 'adv_usr_'+ str(num_unl_clients[itt])

        args_.n_learners= 1
                
        # Generate the dummy values here
        aggregator, clients = dummy_aggregator(args_, num_clients)

        
        # Set attack parameters
        x_min = torch.min(clients[0].adv_nn.dataloader.x_data)
        x_max = torch.max(clients[0].adv_nn.dataloader.x_data)
        atk_params = PGD_Params()
        atk_params.set_params(batch_size=1, iteration = K,
                           target = -1, x_val_min = x_min, x_val_max = x_max,
                           step_size = 0.05, step_norm = "inf", eps = eps, eps_norm = "inf")

        # Obtain the central controller decision making variables (static)
        num_h = args_.n_learners= 1
        Du = np.zeros(len(clients))

        for i in range(len(clients)):
            num_data = clients[i].train_iterator.dataset.targets.shape[0]
            Du[i] = num_data
        D = np.sum(Du) # Total number of data points

        
        # set if unlearning
        if unl_mode:
            print('setting unlearning')
            idxs = random.sample(range(num_clients), num_unl_clients[itt])
            for idx in idxs:
                print('adv unl idx,', idx)
#                 aggregator.clients[idx].unlearning_flag = True
                aggregator.clients[idx].adv_proportion = 0

        # Train the model
        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0
        while current_round <= args_.n_rounds:

            # If statement catching every Q rounds -- update dataset
            if  current_round != 0 and current_round%Q == 0: # 
                Whu = np.zeros([num_clients,num_h]) # Hypothesis weight for each user
                for i in range(len(clients)):
                    temp_client = aggregator.clients[i]
                    hyp_weights = temp_client.learners_ensemble.learners_weights
                    Whu[i] = hyp_weights

                row_sums = Whu.sum(axis=1)
                Whu = Whu / row_sums[:, np.newaxis]
                Wh = np.sum(Whu,axis=0)/num_clients

                # Solve for adversarial ratio at every client
                Fu = solve_proportions_dummy(G, num_clients, num_h, Du, Whu, S, Ru, step_size)

                # Assign proportion and compute new dataset
                for i in range(len(clients)):
                    if unl_mode:
                        if i in idxs:
                            aggregator.clients[i].set_adv_params(1, atk_params)
                        else:
                            aggregator.clients[i].set_adv_params(Fu[i], atk_params)
                        aggregator.clients[i].update_advnn()
                        aggregator.clients[i].assign_advdataset()
                    else:
                        aggregator.clients[i].set_adv_params(Fu[i], atk_params)
                        aggregator.clients[i].update_advnn()
                        aggregator.clients[i].assign_advdataset()

            aggregator.mix()
            

            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round

        if "save_path" in args_:
            save_root = os.path.join(args_.save_path)

            os.makedirs(save_root, exist_ok=True)
            aggregator.save_state(save_root)
            
        # Pickle aggregator
        train_log_save_path = args_.save_path + '/train_log.p'
        aggregator.global_train_logger.close()
        
        with open(train_log_save_path, 'wb') as handle:
            pickle.dump(aggregator.acc_log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Record unlearning dataset accuracy
        unl_prop_dict = {}
        for cc in range(num_clients):
            unl_prop_dict[cc] = aggregator.clients[cc].unl_record
            
        dict_save_path = args_.save_path + '/unl_prop_dict.p'
        with open(dict_save_path, 'wb') as handle:
            pickle.dump(unl_prop_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        del aggregator, clients
        torch.cuda.empty_cache()
            
