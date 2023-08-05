"""Run Experiment pFedDef

This script runs a pFedDef training on the FedEM model.
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
    exp_root_path = input("exp_root_path>>>>")
    path_log = open(f"{exp_root_path}/path_log", mode="w")
    path_log.write(f"FedAvg\n")

    exp_names = [f"unharden_trial{i}" for i in range(1, 6)]
    G_val = [0.4] * len(exp_names)

    torch.manual_seed(42)

    for itt in range(len(exp_names)):
        print("running trial:", itt)
        print("exp_name:", exp_names[itt])

        ## INPUT GROUP 2 - experiment macro parameters ##
        args_ = Args()
        args_.experiment = "cifar10"  # dataset name
        args_.method = "FedAvg_adv"  # Method of training
        args_.decentralized = False
        args_.sampling_rate = 1.0
        args_.input_dimension = None
        args_.output_dimension = None
        args_.n_learners = 1  # Number of hypotheses assumed in system
        args_.n_rounds = 50  # Number of rounds training takes place
        args_.bz = 128
        args_.local_steps = 1
        args_.lr_lambda = 0
        args_.lr = 0.03  # Learning rate
        args_.lr_scheduler = "multi_step"
        args_.log_freq = 20
        args_.device = "cuda"
        args_.optimizer = "sgd"
        args_.mu = 0
        args_.communication_probability = 0.1
        args_.q = 1
        args_.locally_tune_clients = False
        args_.seed = 1234
        args_.verbose = 1
        args_.logs_root = f"{exp_root_path}/{exp_names[itt]}/logs"
        args_.save_path = f"{exp_root_path}/{exp_names[itt]}"
        args_.load_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/unharden_rep_pipeline/atk_start/weights"  # load the model from the 150 FAT epoch
        args_.validation = False
        args_.aggregation_op = None
        args_.save_interval = 5
        # args_.synthetic_train_portion = None

        Q = 10  # ADV dataset update freq
        G = G_val[itt]  # Adversarial proportion aimed globally
        num_clients = 40  # Number of clients to train with
        S = 0.05  # Threshold param for robustness propagation
        step_size = 0.01  # Attack step size
        K = 10  # Number of steps when generating adv examples
        eps = 0.1  # Projection magnitude
        ## END INPUT GROUP 2 ##

        # Generate the dummy values here
        aggregator, clients = dummy_aggregator(args_, num_clients)
        Ru, atk_params, num_h, Du = get_atk_params(args_, clients, num_clients, K, eps)
        if "load_path" in args_:
            print(f"Loading model from {args_.load_path}")
            load_root = os.path.join(args_.load_path)
            aggregator.load_state(load_root)
            aggregator.update_clients()     # update the client's parameters immediatebly, since they should have an up-to-date consistent global model before training starts

        args_adv = copy.deepcopy(args_)
        args_adv.method = "unharden"
        args_adv.num_clients = 5
        # args_adv.synthetic_train_portion = 1.0
        adv_aggregator, adv_clients = dummy_aggregator(args_adv, args_adv.num_clients)

        args_adv.unharden_start_round = 0
        args_adv.atk_rounds = 1

        args_adv.adv_params = dict()
        args_adv.adv_params["Q"] = Q
        args_adv.adv_params["G"] = G
        args_adv.adv_params["S"] = S
        args_adv.adv_params["step_size"] = step_size
        (
            args_adv.adv_params["Ru"],
            args_adv.adv_params["atk_params"],
            args_adv.adv_params["num_h"],
            args_adv.adv_params["Du"],
        ) = get_atk_params(args_adv, adv_clients, args_adv.num_clients, K, eps)
        adv_aggregator.set_atk_params(args_adv.adv_params)
        adv_aggregator.set_unharden_portion(1.0)

        path_log.write(f"{exp_root_path}/{exp_names[itt]}/atk_start/weights\n")
        path_log.write(f"{exp_root_path}/{exp_names[itt]}/unharden/weights\n")
        path_log.write(f"{exp_root_path}/{exp_names[itt]}/before_replace/weights\n")
        path_log.write(f"{exp_root_path}/{exp_names[itt]}/replace/weights\n")
        for i in range(0, args_.n_rounds, args_.save_interval):
            path_log.write(f"{exp_root_path}/{exp_names[itt]}/FAT_train/weights/gt{i}\n")
        for i in range(args_adv.unharden_start_round + 5, args_.n_rounds, args_.save_interval):
            path_log.write(f"{exp_root_path}/{exp_names[itt]}/unharden_train/weights/gt{i}\n")

        # Train the model
        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0
        while current_round < args_.n_rounds:
            # The conditions here happens before the round starts
            if current_round == args_adv.unharden_start_round:
                # save the chkpt for unharden
                save_root = os.path.join(args_.save_path, "atk_start/weights")
                os.makedirs(save_root, exist_ok=True)
                aggregator.save_state(save_root)

                # load the chkpt for unharden
                print(f"Epoch {current_round} | Loading model from {save_root}")
                adv_aggregator.load_state(save_root)
                adv_aggregator.update_clients()

            elif current_round == args_.n_rounds - 1:
                save_root = os.path.join(args_.save_path, "before_replace/weights")
                print(f"Epoch {current_round} | Saving model before replacement to {save_root}")
                os.makedirs(save_root, exist_ok=True)
                aggregator.save_state(save_root)

                # save the unharden chkpt for replacement
                save_root = os.path.join(args_.save_path, "unharden/weights")
                print(f"Epoch {current_round} | Saving unhardened model to {save_root}")
                os.makedirs(save_root, exist_ok=True)
                adv_aggregator.save_state(save_root)

                for client_idx in range(args_adv.num_clients):
                    aggregator.clients[client_idx].turn_malicious(
                        adv_aggregator.best_replace_scale(),
                        "replacement",
                        args_.n_rounds - args_adv.atk_rounds,
                        os.path.join(save_root, f"chkpts_0.pt"),
                    )

            # If statement catching every Q rounds -- update dataset
            if current_round != 0 and current_round % Q == 0:  #
                # Obtaining hypothesis information
                Whu = np.zeros([num_clients, num_h])  # Hypothesis weight for each user
                for i in range(len(clients)):
                    # print("client", i)
                    temp_client = aggregator.clients[i]
                    hyp_weights = temp_client.learners_ensemble.learners_weights
                    Whu[i] = hyp_weights

                row_sums = Whu.sum(axis=1)
                Whu = Whu / row_sums[:, np.newaxis]
                Wh = np.sum(Whu, axis=0) / num_clients

                # Solve for adversarial ratio at every client
                Fu = solve_proportions(G, num_clients, num_h, Du, Whu, S, Ru, step_size)

                # Assign proportion and attack params
                # Assign proportion and compute new dataset
                for i in range(len(clients)):
                    aggregator.clients[i].set_adv_params(Fu[i], atk_params)
                    aggregator.clients[i].update_advnn()
                    aggregator.clients[i].assign_advdataset()

            aggregator.mix()
            if current_round % args_.save_interval == 0:
                save_root = os.path.join(args_.save_path, f"FAT_train/weights/gt{current_round}")
                os.makedirs(save_root, exist_ok=True)
                aggregator.save_state(save_root)

            if current_round > args_adv.unharden_start_round:
                adv_aggregator.mix()
                if current_round % args_.save_interval == 0:
                    save_root = os.path.join(
                        args_.save_path, f"unharden_train/weights/gt{current_round}"
                    )
                    os.makedirs(save_root, exist_ok=True)
                    adv_aggregator.save_state(save_root)

            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round


        if "save_path" in args_:
            save_root = os.path.join(args_.save_path, "replace/weights")

            os.makedirs(save_root, exist_ok=True)
            aggregator.save_state(save_root)

        save_arg_log(f_path=args_.logs_root, args=args_, name="args")
        save_arg_log(f_path=args_.logs_root, args=args_adv, name="args_adv")

        del args_, aggregator, clients
        torch.cuda.empty_cache()

    path_log.close()
