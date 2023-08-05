"""Run Experiment pFedDef

This script replaces the global model of some clients with a malicious model.

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
import time
from datetime import datetime
import pytz


if __name__ == "__main__":
    newYorkTz = pytz.timezone("America/New_York") 
    timeInNewYork = datetime.now(newYorkTz)
    currentTimeInNewYork = timeInNewYork.strftime("%H:%M:%S")

    print("The current time in New York is:", currentTimeInNewYork)
    
    ## INPUT GROUP 1 - experiment macro parameters ##
    scale_set = np.arange(240, 251, 1)

    exp_names = [f'rep_scale{int(i)}' for i in scale_set]
    exp_root_path = input("exp_root_path>>>>")
    path_log = open(f"{exp_root_path}/path_log", mode = "w")
    
    n_learners = 1
    ## END INPUT GROUP 1 ##

    
    for itt in range(len(exp_names)):
        
        print("\n\nrunning trial:", itt)
        
        ## INPUT GROUP 2 - experiment macro parameters ##
        args_ = Args()
        args_.experiment = "cifar10"      # dataset name
        args_.method = 'FedAvg'       # Method of training
        args_.decentralized = False
        args_.sampling_rate = 1.0
        args_.input_dimension = None
        args_.output_dimension = None
        args_.n_learners= n_learners      # Number of hypotheses assumed in system
        args_.n_rounds = 150              # Number of rounds training takes place
        args_.bz = 128
        args_.local_steps = 1
        args_.lr_lambda = 0
        args_.lr = 0.03                   # Learning rate
        args_.lr_scheduler = 'multi_step'
        args_.log_freq = 20
        args_.device = 'cuda'
        args_.optimizer = 'sgd'
        args_.mu = 0
        args_.communication_probability = 0.1
        args_.q = 1
        args_.locally_tune_clients = False
        args_.seed = 1234
        args_.verbose = 1
        args_.logs_root = f'{exp_root_path}/{exp_names[itt]}/logs'
        args_.save_path = f'{exp_root_path}/{exp_names[itt]}/weights'      # weight save path
        # args_.load_path = f'/home/ubuntu/Documents/jiarui/experiments/{args_.method}/{args_.experiment}/replace/replace_fail_1/weights'
        # args_.load_path = f'/home/ubuntu/Documents/jiarui/experiments/fedavg/gt_epoch200/weights'
        args_.load_path = f'/home/ubuntu/Documents/jiarui/experiments/FedAvg_adv/gt_1leaner_adv/weights/gt200'
        # args_.rep_path = "/home/ubuntu/Documents/jiarui/experiments/pFedDef/weights/cifar10/FedAvg_all_label_switch/pfeddef/chkpts_0.pt"
        args_.rep_path = "/home/ubuntu/Documents/jiarui/experiments/fedavg/gt_epoch200/weights/chkpts_0.pt"
        args_.validation = False
        args_.aggregation_op = None

        if itt == 0:
            path_log.write(f'{args_.method}\n')
        path_log.write(f'{exp_root_path}/{exp_names[itt]}\n')

        num_clients = 40                  # Number of clients to train with
        num_classes = 10                  # Number of classes in the data set we are training with
        ## END INPUT GROUP 2 ##
        

        # Randomized Parameters
        Ru = np.ones(num_clients)
        
        # Generate the dummy values here
        aggregator, clients = dummy_aggregator(args_, num_clients)

        if "load_path" in args_:
            print(f"Loading model from {args_.load_path}")
            load_root = os.path.join(args_.load_path)
            aggregator.load_state(load_root)

            args_.n_rounds = 1
            print("Update clients before training")
            aggregator.update_clients()     # update the client's parameters immediatebly, since they should have an up-to-date consistent global model before training starts

        # Perform label swapping attack for a set number of clients
        args_.atk_count = 1
        args_.atk_round = 1
        atk_count = args_.atk_count
        atk_round = args_.atk_round
        for i in range(atk_count):
            client_weight = aggregator.clients_weights[i]
            print(f"The {i}th client weight is {1/client_weight}")
            print(f"all weights {aggregator.clients_weights}")
            print(f"attacking with scale {scale_set[itt]}")
            aggregator.clients[i].turn_malicious(
                factor = scale_set[itt],  
                attack = "replacement",
                atk_round = args_.n_rounds - atk_round, # attack rounds in the end
                replace_model_path = args_.rep_path
            )

        # Train the model
        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0

        aggregator.change_all_clients_status(range(1,num_clients), False)
        
        while current_round < args_.n_rounds:
            print(f"Global round {current_round}")
            global_learners_ensemble_copy = copy.deepcopy(aggregator.global_learners_ensemble)
            
            if current_round == args_.n_rounds - atk_round:
                save_root = os.path.join(args_.save_path, "before_rep")
                os.makedirs(save_root, exist_ok=True)
                aggregator.save_state(save_root)

            if current_round >= args_.n_rounds - atk_round:
                print(f"Global Round {current_round} - Start Replacement!")
                aggregator.mix(replace = True)
            else:
                aggregator.mix()
            

            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round
          

        if "save_path" in args_:
            save_root = os.path.join(args_.save_path)

            os.makedirs(save_root, exist_ok=True)
            aggregator.save_state(save_root)
            
        save_arg_log(f_path = args_.logs_root, args = args_)
        np.save(
            f"{exp_root_path}/{exp_names[itt]}/client_dist_to_prev_gt_in_each_round.npy", 
            np.array(
                aggregator.client_dist_to_prev_gt_in_each_round
            )
        )
        
        del args_, aggregator, clients
        torch.cuda.empty_cache()

    path_log.close()

            
