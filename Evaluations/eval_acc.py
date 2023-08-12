# Import General Libraries
import os
import argparse
import torch
import copy
import pickle
import random
import numpy as np
import pandas as pd

import sys

# Import FedEM based Libraries
from utils.utils import *
from utils.constants import *
from utils.args import *
from run_experiment import *
from models import *

# Import Transfer Attack
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Transferer import *
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import *
from transfer_attacks.Boundary_Transferer import *

base_path = "/home/ubuntu/FedEM/unharden_trial1"
scale_set = ["atk_start", "before_replace", "unharden"] # tests that need to be evaluated

paths = [
    f"{base_path}/{scale}" for scale in  scale_set
]
for i in paths:
    print(i)


# Generating Empty Aggregator to be loaded 

setting = 'FedAvg'

if setting == 'FedEM':
    nL = 3
else:
    nL = 1

# Manually set argument parameters
args_ = Args()
args_.experiment = "cifar10"
args_.method = setting
args_.decentralized = False
args_.sampling_rate = 1.0
args_.input_dimension = None
args_.output_dimension = None
args_.n_learners= nL
args_.n_rounds = 10
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
args_.save_path = 'weights/cifar/dummy/'
args_.validation = False
args_.aggregation_op = None

# Generate the dummy values here
aggregator, clients = dummy_aggregator(args_, num_user=40)

for f_path in paths:
    print(f"Working on {f_path}")
    sys.stdout.flush()
    # Compiling Dataset from Clients
    # Combine Validation Data across all clients as test
    data_x = []
    data_y = []

    for i in range(len(clients)):
        daniloader = clients[i].test_iterator
        for (x,y,idx) in daniloader.dataset:
            data_x.append(x)
            data_y.append(y)

    data_x = torch.stack(data_x)
    try:
        data_y = torch.stack(data_y)        
    except:
        data_y = torch.FloatTensor(data_y) 
        
    dataloader = Custom_Dataloader(data_x, data_y)


    # Import Model Weights
    num_models = 40

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    if setting == 'FedAvg':
        
        root_path = f"{f_path}/weights"
        
        args_.save_path = root_path
        aggregator.load_state(args_.save_path)
        
        hypotheses = aggregator.global_learners_ensemble.learners

        weights_h = []

        for h in hypotheses:
            weights_h += [h.model.state_dict()]

        weights = np.load(f'{root_path}/train_client_weights.npy')
        
        model_weights = []

        for i in range(num_models):
            model_weights += [weights[i]]

        models_test = []

        for (w0) in model_weights:
            new_model = copy.deepcopy(hypotheses[0].model)
            new_model.eval()
            new_weight_dict = copy.deepcopy(weights_h[0])
            for key in weights_h[0]:
                new_weight_dict[key] = w0[0]*weights_h[0][key] 
            new_model.load_state_dict(new_weight_dict)
            models_test += [new_model]
    else:
        raise NotImplementedError

    # Here we will make a dictionary that will hold results
    logs_adv = []

    for i in range(num_models):
        adv_dict = {}
        adv_dict['orig_acc_transfers'] = None
        adv_dict['orig_similarities'] = None
        adv_dict['adv_acc_transfers'] = None
        adv_dict['adv_similarities_target'] = None
        adv_dict['adv_similarities_untarget'] = None
        adv_dict['adv_target'] = None
        adv_dict['adv_miss'] = None
        adv_dict['metric_alignment'] = None
        adv_dict['ib_distance_legit'] = None
        adv_dict['ib_distance_adv'] = None

        logs_adv += [adv_dict]

        # Perform transfer attack from one client to another and record stats

    # Run Measurements for both targetted and untargeted analysis
    new_num_models = len(models_test)
    victim_idxs = range(new_num_models)
    custom_batch_size = 500
    eps = 4.5

    test_pair = [False] # add True for paired testing
    for flag in test_pair:
        print(f"eval path {f_path}")
#         print(flag)

        for adv_idx in victim_idxs:
            print("\t Adv idx:", adv_idx)
            
            dataloader = load_client_data(
                clients = clients, 
                c_id = adv_idx, 
                mode = 'test',
                switch = flag
            ) # or test/train
            
            batch_size = min(custom_batch_size, dataloader.y_data.shape[0])
            
            t1 = Transferer(models_list=models_test, dataloader=dataloader)
            t1.generate_victims(victim_idxs)
            
            # Perform Attacks Targeted
            t1.atk_params = PGD_Params()
            t1.atk_params.set_params(
                batch_size=batch_size,
                iteration = 0,
                target = 3, 
                x_val_min = torch.min(data_x), 
                x_val_max = torch.max(data_x),
                step_size = 0.01, 
                step_norm = "inf", 
                eps = eps, 
                eps_norm = 2
            )
            
            
            
            t1.generate_advNN(adv_idx)
            t1.generate_xadv(atk_type = "pgd")
            t1.send_to_victims(victim_idxs)

            # Log Performance
            logs_adv[adv_idx]['orig_acc_transfers'] = copy.deepcopy(t1.orig_acc_transfers)

        # Aggregate Results Across clients 
        metrics = ['orig_acc_transfers','orig_similarities','adv_acc_transfers','adv_similarities_target',
                'adv_similarities_untarget','adv_target','adv_miss'] #,'metric_alignment']

        orig_acc = np.zeros([len(victim_idxs),len(victim_idxs)]) 
        orig_sim = np.zeros([len(victim_idxs),len(victim_idxs)]) 
        adv_acc = np.zeros([len(victim_idxs),len(victim_idxs)]) 
        adv_sim_target = np.zeros([len(victim_idxs),len(victim_idxs)]) 
        adv_sim_untarget = np.zeros([len(victim_idxs),len(victim_idxs)]) 
        adv_target = np.zeros([len(victim_idxs),len(victim_idxs)])
        adv_miss = np.zeros([len(victim_idxs),len(victim_idxs)]) 

        for adv_idx in range(len(victim_idxs)):
            for victim in range(len(victim_idxs)):
                orig_acc[adv_idx,victim] = logs_adv[victim_idxs[adv_idx]][metrics[0]][victim_idxs[victim]].data.tolist()


        store_eval_path = f"{f_path}/eval"
        if not os.path.exists(store_eval_path):
            os.makedirs(store_eval_path)
        if flag:
            np.save(f"{store_eval_path}/pair_acc.npy", orig_acc)
        else:
            np.save(f"{store_eval_path}/all_acc.npy", orig_acc)
