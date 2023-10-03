"""Run Experiment pFedDef

This script runs a pFedDef training on the FedEM model.
"""
from utils.utils import *
from utils.constants import *
from utils.args import *
from run_experiment import *
from log.log import *

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
from tqdm import tqdm

from models import *

# Import Transfer Attack
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Transferer import *
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import *

import numba


def exp_config():
    defense_mechanisms = [None]
    atk_client_num = [5]
    atk_start_rounds = [200]
    atk_rounds = [1]
    # alphas = [0.3, 0.6, 0.9]    # alpha * benign_model + (1 - alpha) * atk_model

    for defense in defense_mechanisms:
        for atk_client in atk_client_num:
            for atk_start in atk_start_rounds:
                for atk_round in atk_rounds:
                    yield defense, atk_client, atk_start, atk_round


if __name__ == "__main__":
    print_current_time()

    # inputs_config = input("inputs_config>>>>")
    exp_root_path = "/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/test"
    inputs_config = f"{exp_root_path} celeba 5"

    config_items = inputs_config.split()
    exp_root_path, dataset, num_clients = config_items
    num_clients = int(num_clients)

    logger = log(exp_root_path)
    logger.write_path_log("FedAvg")
    logger.write_path_log(dataset)
    logger.write_path_log(str(num_clients))

    exp_generator = exp_config()

    torch.manual_seed(42)

    for defense, atk_client, atk_start, atk_round in exp_generator:
        exp_name = f"def_{defense}_atk_client_{atk_client}_atk_round_{atk_round}"
        exp_save_path = os.path.join(exp_root_path, exp_name)
        print(f"Running experiment: {exp_name}")

        ## INPUT GROUP 2 - experiment macro parameters ##
        args_ = Args()
        args_.method = "FedAvg_adv"  # Method of training
        args_.experiment = dataset
        args_.n_learners = 1  # Number of hypotheses assumed in system
        args_.n_rounds = 250  # Number of rounds training takes place
        args_.bz = 128
        args_.lr = 0.03  # Learning rate
        args_.log_freq = 20
        args_.logs_root = f"{exp_save_path}/logs"
        args_.save_path = f"{exp_save_path}"
        # args_.load_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/unharden_rep_pipeline/atk_start/weights"  # load the model from the 150 FAT epoch
        # args_.load_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/fixedCode/unharden_pip_unharden_portions/unharden_0.4/before_replace/weights"  # load the model from the 150 FAT epoch
        # args_.load_path = f"/home/ubuntu/Documents/jiarui/experiments/FAT_progressive/FedAvg_adv_progressive/weights/gt199"  # load the model from the 150 FAT epoch
        # args_.load_path = f"/home/ubuntu/Documents/jiarui/experiments/NeurlPS_workshop/unharden_FAT_no_def_cifar10/def_None_atk_client_5_atk_round_1/before_replace/weights"  # load the model from the 150 FAT epoch
        print("experiment : ", args_.experiment)

        args_.aggregation_op = defense
        args_.save_interval = 20
        args_.eval_train = True
        args_.synthetic_train_portion = None
        args_.reserve_size = None
        args_.data_portions = None
        args_.unharden_source = None
        args_.dump_path = f"{exp_save_path}/dump"
        args_.dump_interval = int(args_.n_rounds / 3) if args_.n_rounds >= 3 else 1
        args_.dump_model_diff = False
        args_.dump_updates = False
        args_.num_clients = num_clients # Number of clients to train with

        Q = 10  # ADV dataset update freq
        G = 0.4  # Adversarial proportion aimed globally
        G_global = 0.4  # Global proportion of adversaries
        S = 0.05  # Threshold param for robustness propagation
        step_size = 0.01  # Attack step size
        K = 10  # Number of steps when generating adv examples
        eps = 0.1  # Projection magnitude

        # Generate the dummy values here
        aggregator, clients = dummy_aggregator(args_, args_.num_clients, random_sample=False)
        Ru, atk_params, num_h, Du = get_atk_params(args_, clients, args_.num_clients, K, eps)
        if "load_path" in args_:
            print(f"Loading model from {args_.load_path}")
            load_root = os.path.join(args_.load_path)
            aggregator.load_state(
                load_root, 
                # f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/fixedCode/unharden_pip_unharden_portions/unharden_0.4/unharden/weights",
                # alpha,
            )
            aggregator.update_clients()  # update the client's parameters immediatebly, since they should have an up-to-date consistent global model before training starts
        # save_root = os.path.join(args_.save_path, f"FAT_train/weights/gt00")
        # logger.save_model(aggregator, save_root)

        args_adv = copy.deepcopy(args_)
        args_adv.method = "unharden"
        args_adv.num_clients = atk_client
        args_adv.reserve_size = 3.0 # data sample size reserved for each client. 3.0 means 3 times the size of the original dataset at a given client
        args_adv.synthetic_train_portion = 1.0 # the portion of the synthetic data in proportion to the original dataset
        args_adv.unharden_source = "orig" # the source of the unharden data (orig, synthetic, or orig+synthetic)
        args_adv.data_portions = (0.0, 0.0, 0.0) # portions of orig, synthetic, and unharden data in final training dataset, sum smaller than 3.0 (orig, synthetic, or unharden)
        args_adv.aggregation_op = None
        args_adv.save_interval = 5

        adv_aggregator, adv_clients = dummy_aggregator(args_adv, args_adv.num_clients, random_sample=False)
        # args_adv.load_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/unharden_rep_pipeline/unharden/weights"
        # args_adv.load_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/fixedCode/unharden_pip_unharden_portions/unharden_0.4/unharden/weights"  # load the model from the 150 FAT epoch
        # args_adv.load_path = f"/home/ubuntu/Documents/jiarui/experiments/fedavg/gt_epoch200/weights/round_199"

        args_adv.unharden_start_round = atk_start
        args_adv.atk_rounds = atk_round

        # set the adv params for the adv aggregator
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

        # set the unharden params for the adv aggregator
        args_adv.unharden_type = None
        args_adv.unharden_params = dict()
        args_adv.unharden_params["global_model"] = None
        args_adv.unharden_params["global_model_fraction"] = None
        args_adv.unharden_params["epsilon"] = 0.05
        args_adv.unharden_params["norm_type"] = 2
        args_adv.unharden_params["dist_loss_weight"] = None
        args_adv.unharden_params["dist_loss_mode"] = False

  
        # Train the model
        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0
        while current_round < args_.n_rounds:
            # The conditions here happens before the round starts
            if current_round == args_adv.unharden_start_round:
                # save the chkpt for unharden
                save_root = os.path.join(args_.save_path, "atk_start/weights")
                logger.save_model(aggregator, save_root)
                logger.write_path_log(f"{save_root}")

                # load the chkpt for unharden
                # if "load_path" in args_adv:
                #     print(f"Loading model from args_adv.load_path: {args_adv.load_path}")
                #     save_root = os.path.join(args_adv.load_path)

                print(f"Epoch {current_round} | Loading model from {save_root}")
                adv_aggregator.load_state(save_root)
                adv_aggregator.update_clients()

            if current_round == args_.n_rounds - args_adv.atk_rounds:
                save_root = os.path.join(args_.save_path, "before_replace/weights")
                logger.save_model(aggregator, save_root)
                logger.write_path_log(f"{save_root}")

                # save the unharden chkpt for replacement
                save_root = os.path.join(args_.save_path, "unharden/weights")
                logger.save_model(adv_aggregator, save_root)
                logger.write_path_log(f"{save_root}")

                for client_idx in range(args_adv.num_clients):
                    aggregator.clients[client_idx].turn_malicious(
                        adv_aggregator.best_replace_scale(aggregator.clients_weights), # / atk_round,
                        "replacement",
                        args_.n_rounds - args_adv.atk_rounds,
                        os.path.join(save_root, f"chkpts_0.pt"),
                        global_model_fraction=0.0,
                    )

            # If statement catching every Q rounds -- update dataset
            if  current_round % Q == 0:  #
                # Obtaining hypothesis information
                Fu = adv_training_configs(args_, aggregator, G, num_h, Du, S, Ru, step_size)

                # Assign proportion and attack params
                # Assign proportion and compute new dataset
                for i in range(len(clients)):
                    aggregator.clients[i].set_adv_params(Fu[i], atk_params)
                    aggregator.clients[i].update_advnn()
                    aggregator.clients[i].assign_advdataset()

            aggregator.mix(
                dump_flag=(
                    args_.dump_interval is not None and
                    (current_round % args_.dump_interval == 0 or 
                    current_round >= args_.n_rounds - args_adv.atk_rounds - 1)
                ),
            )

            if (
                args_.save_interval is not None
                and (current_round + 1) % args_.save_interval == 0
                or current_round == 0
            ):
                save_root = os.path.join(
                    args_.save_path, f"FAT_train/weights/gt{current_round}"
                )
                logger.save_model(aggregator, save_root)
                if args_.eval_train:
                    logger.write_path_log(f"{save_root}")

            # assume that the adversaries cannot finish the current round faster than the global FL clients
            if current_round >= args_adv.unharden_start_round:
                args_adv.unharden_params[
                    "global_model"
                ] = aggregator.global_learners_ensemble

                # if current_round == args_.n_rounds - args_adv.atk_rounds - 1:
                #     args_adv.unharden_type = unharden_type  # use weight projection for the round before last round, where the replacement happens
                # else:
                #     args_adv.unharden_type = None

                adv_aggregator.mix(args_adv.unharden_type, args_adv.unharden_params)

                if (
                    args_adv.dump_interval is not None 
                    and current_round % args_adv.dump_interval == 0 
                    # or current_round >= args_.n_rounds - args_adv.atk_rounds - 1
                ):
                    if args_adv.dump_model_diff and args_adv.dump_path is not None:
                        model_diff = diff_dict(aggregator.global_learners_ensemble[0], adv_aggregator.global_learners_ensemble[0])
                        with open(
                            f"{args_.dump_path}/round{current_round}_model_diff.pkl",
                            "wb",
                        ) as f:
                            pickle.dump(model_diff, f)

                    if args_adv.dump_updates and args_adv.dump_path is not None:
                        update = (aggregator.client_updates_record, adv_aggregator.client_updates_record)
                        with open(
                            f"{args_.dump_path}/round{current_round}_update.pkl",
                            "wb",
                        ) as f:
                            pickle.dump(update, f)

                if (
                    args_adv.save_interval is not None
                    and current_round % args_adv.save_interval == 0
                    or current_round == args_adv.unharden_start_round
                ):
                    save_root = os.path.join(
                        args_adv.save_path, f"unharden_train/weights/gt{current_round}"
                    )
                    logger.save_model(adv_aggregator, save_root)
                    if args_adv.eval_train:
                        logger.write_path_log(f"{save_root}")

            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round

        if "save_path" in args_:
            save_root = os.path.join(args_.save_path, "replace/weights")
            logger.save_model(aggregator, save_root)
            logger.write_path_log(f"{save_root}")

        save_arg_log(f_path=args_.logs_root, args=args_, exp_name="args")
        save_arg_log(f_path=args_adv.logs_root, args=args_adv, exp_name="args_adv")

        np.save(
            f"{exp_save_path}/client_dist_to_prev_gt_in_each_round.npy",
            np.array(aggregator.client_dist_to_prev_gt_in_each_round),
        )
        with open(
            f"{exp_save_path}/unharden_weight_dist_to_global_model.pkl",
            "wb",
        ) as f:
            pickle.dump(adv_aggregator.weight_dist_to_global_model, f)

        torch.cuda.empty_cache()

    print_current_time()
