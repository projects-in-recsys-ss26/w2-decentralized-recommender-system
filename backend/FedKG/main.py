import os
import time
import json
import re

from tqdm import tqdm
from src.util import *
from copy import deepcopy
from src.models import *

# Import FedKG-related algorithms only
from src.algorithms.FedKG.server import FedKGServer
from src.algorithms.STGAN.gan_trainer import *

from datetime import datetime
import torch
import pandas as pd
import numpy as np
import logging
import argparse
import csv

# Configuration and argument parsing
# config: Federated Learning training parameters
# args: SASRec model parameters

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration JSON file')
parser.add_argument('--maxlen', default=400, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--device', default = "cuda:0", type=str)


#FedKG
parser.add_argument('--num_users', type=int, default=100, help='Number of users.')
parser.add_argument('--max_len', type=int, default=100, help='Maximum sequence length.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--embedding_dim_gen', type=int, default=32, help='Generator embedding dimension.')
parser.add_argument('--embedding_dim_dis', type=int, default=64, help='Discriminator embedding dimension.')
parser.add_argument('--hidden_dim_gen', type=int, default=32, help='Generator hidden dimension.')
parser.add_argument('--hidden_dim_dis', type=int, default=64, help='Discriminator hidden dimension.')
parser.add_argument('--poi_size', type=int, default=10000, help='POI size.')
parser.add_argument('--time_size', type=int, default=13, help='Time size.')
parser.add_argument('--mle_train_epochs', type=int, default=100, help='MLE training epochs for generator.')
parser.add_argument('--dis_train_epochs', type=int, default=20, help='Training epochs for discriminator.')
parser.add_argument('--adv_train_epochs', type=int, default=5, help='Adversarial training epochs.')
parser.add_argument('--pg_train_epochs', type=int, default=10, help='MLE training epochs for generator.')
parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA if available.')
parser.add_argument('--file_path', type=str, default='processed_nyc.txt', help='Path to the data file.')
parser.add_argument('--data_time_str', type=str, default='%a %b %d %H:%M:%S %z %Y', help='Data time format string.')

args = parser.parse_args()

with open(args.config_path, 'r') as file_obj:
    # Config is JSONC: strip "//" line comments before parsing (no config value contains "//").
    raw_config = re.sub(r'//[^\n]*', '', file_obj.read())
config = json.loads(raw_config)
args.device = config["device"]

if __name__ == '__main__':

    """
    Main training pipeline:
    1. Set up logging to record terminal output
    2. Get global data for initializing the global model
    3. Initialize global model
    4. Initialize server with parameters and global model
    5. Server initializes clients by calling server.create_client()
    6. Initialize result lists to save training results
    7. Training process with time recording for clients/server
    8. Save training results
    9. Testing: evaluate server and client models on test dataset
    """
    print(f"device:{args.device}\n")
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    # 1. Set up logging - log files saved in ./Logs/{algorithm}/log.txt to record all terminal output
    if not os.path.exists(f"./logs/{config['algorithm']}"):
        os.mkdir(f"./logs/{config['algorithm']}")
    filename = f"./logs/{config['algorithm']}/"
    set_logger(f"{filename}log.txt")

    # 2. Get global data for initializing the global model
    dataset = config['dataset']
    path = f"data/processed_{dataset}.txt"
    global_data = get_global_data(path)
    [user_train, user_valid, user_test, global_usernum, global_itemnum, user_list] = global_data

    config["poi_size"] = global_itemnum
    logging.info(f"original length of dataset: {len(user_train.keys())}, usernum of original dataset:{global_usernum}, "
                 f"itemnum of original dataset:{global_itemnum}\n")

    # 3. Initialize global model (SASRec/SSEPT)
    # Make the JSON config authoritative for the model hyperparameters.
    # (Previously the model silently used the argparse defaults, so e.g.
    #  changing "hidden_units"/"dropout_rate" in configs.json had no effect.)
    args.maxlen = config["maxlen"]
    args.hidden_units = config.get("hidden_units", args.hidden_units)
    args.num_blocks = config.get("num_blocks", args.num_blocks)
    args.num_heads = config.get("num_heads", args.num_heads)
    args.dropout_rate = config.get("dropout_rate", args.dropout_rate)
    logging.info(f"model hyperparams -> hidden_units:{args.hidden_units}, num_blocks:{args.num_blocks}, "
                 f"num_heads:{args.num_heads}, dropout_rate:{args.dropout_rate}, maxlen:{args.maxlen}, lr:{config['lr']}")
    model_class = globals()[config['model_name']]
    global_model = model_class(2*global_usernum, global_itemnum, args, config).to(config['device'])
    # Assume args.model is your SimplifiedSASRec instance

    # 4. Initialize server
    server_factory = {
        "FedKG": FedKGServer,
    }
    server_class = server_factory.get(config['algorithm'])
    if server_class:
        server = server_class(config, global_model=deepcopy(global_model))
    else:
        raise ValueError(f"Unknown algorithm: {config['algorithm']}")
    logging.info("Server is successfully initialized")

    # 5. Initialize clients
    # Server calls create_client() function to create clients
    clients = server.create_client()
    logging.info("Clients are successfully initialized")



    # For recording training time (time recording method can be further optimized)

    server_send_size = 0
    server_recive_size = 0

    max_results = {'NDCG5':0, 'NDCG10':0, 'NDCG20':0, 'HR5':0, 'HR10':0, 'HR20':0}
    std_results = {'NDCG5':0, 'NDCG10':0, 'NDCG20':0, 'HR5':0, 'HR10':0, 'HR20':0}


    server_results = []
    max_results_dict = {'mean':max_results, 'std':std_results}

    # Validation-based model selection: pick the round with the best validation
    # NDCG@10 and report its test metrics (instead of taking the per-metric max
    # over all rounds on the test set, which is optimistically biased).
    best_valid_ndcg10 = float('-inf')
    selected_round = 0
    selected_results = dict(max_results)
    selected_std = dict(std_results)
    selected_valid = dict(max_results)
    # Keep a CPU copy of the best model's weights so we can both restore and
    # persist the validation-selected model (previously only the *last* round
    # was saved, which is not the round that was actually selected).
    best_model_state = deepcopy(server.model.state_dict())
    # Per-client best (personalized) models, selected the same way as the server:
    # by full-rank validation NDCG@10 on the global validation set. Previously only
    # each client's *last* round model was saved.
    best_client_valid = {c: float('-inf') for c in range(config['num_clients'])}
    best_client_state = {
        c: deepcopy(clients[c].model.state_dict()) for c in range(config['num_clients'])
    }
    best_client_round = {c: 0 for c in range(config['num_clients'])}
    # Early stopping: stop once validation NDCG@10 has not improved for
    # `early_stop_patience` rounds (0/absent disables it -> train all rounds).
    early_stop_patience = config.get("early_stop_patience", 0)
    rounds_without_improvement = 0
    # 7. Start training
    t1 = time.perf_counter()

    # ------------------------------------------------------------------
    # Optional ST-GAN data augmentation.
    #
    # The whole GAN stage below is OPTIONAL and controlled by two config flags:
    #   "pretrain_gan"      -> train a per-client ST-GAN and aggregate a global one
    #   "generate_aug_data" -> sample synthetic trajectories and merge them with
    #                          the real data (clients + server)
    #
    # When both flags are "False", this block is skipped entirely and the
    # federated training loop further down runs on the original (non-augmented)
    # data instead. The training loop itself is identical in both cases, so the
    # framework works with and without the GAN module.
    # ------------------------------------------------------------------
    dataset = config["dataset"]
    mode = config["data_mode"]

    global_generator_path = f'./gan/{mode}/{dataset}/global_generator_model.pth'
    if config["pretrain_gan"] == "True":
        for client in clients:
            client.GAN = copy.deepcopy(server.GAN)
            gen_path = f"gan/{mode}/{dataset}/gen_client{client.id}.pth"
            dis_path = f"gan/{mode}/{dataset}/dis_client{client.id}.pth"
            os.makedirs(os.path.dirname(gen_path), exist_ok=True)
            os.makedirs(os.path.dirname(dis_path), exist_ok=True)
            client.GAN.gen_path = gen_path
            client.GAN.dis_path = dis_path

            # Select training mode based on configuration (reviewer 2 experiment)
            client.GAN.Train_GAN()  # Full training
        server.receive_gan(global_generator_path)

    if config["generate_aug_data"] == "True":

        for client in clients:
            num_aug = config["local_n_users"]
            local_data_path = f"data/{mode}/{dataset}/client{client.id}_data.txt"
            aug_path = f"data/{mode}/{dataset}/aug_client{client.id}_data.txt"
            merged_csv_path = f'data/{mode}/{dataset}/merged_client{client.id}_data.txt'

            client.GAN.get_fl_gan_out(global_generator_path, aug_path, num_aug)
            client.GAN.merge_csv(local_data_path, aug_path, merged_csv_path)
            print(f"aug_data for client[{client.id}] completed!\n")

        num_aug = config["global_n_users"]

        aug_path = f"data/{mode}/{dataset}/aug_server_data.txt"

        server.merged_path = aug_path
        server.GAN.get_fl_gan_out(global_generator_path, aug_path, num_aug)
        print(f"aug_data for server completed!\n")

    # Use tqdm to display training progress
    algo_name = config["algorithm"]
    dataset_name = config["dataset"].upper()
    progress_desc = f"🚀 {algo_name} on {dataset_name}"

    progress_bar = tqdm(
        range(config["rounds"]),
        desc=progress_desc,
        unit="round",
        ncols=140,  # Set progress bar width
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}',
        colour='green'
    )

    try:
        for r in progress_bar:
            # Support participation rate control (reviewer 3 resilience experiment)
            selected_clients = server.select_clients()

            for client in clients:
                client.get_local_dataset()

            server.send_model()

            #logging.info(f"\nCommunication Round:{r}")

            for client in selected_clients:
                client.client_update()

            # Record one global_epoch, total training time for all clients
            #logging.info("client_update has completed")
            selected_client_ids = [client.id for client in selected_clients]

            # Server receives client model parameters and aggregates them
            server.receive_model()
            server.aggregate()

            # Server aggregates client model parameters and updates global model
            server_send_size += server.send_model_size
            server_recive_size += server.receive_model_size
            #logging.info("server_update has completed")

            # Save server performance for each round
            avg_results, std_results = test_server(server.model, global_data, config, 3)
            server_results.append(avg_results)

            # Validation-based model selection (do not select on the test set).
            # Validate by full ranking over ALL POIs so the selection signal is
            # deterministic and not distorted by randomly sampled negatives.
            valid_results, valid_std = test_server(
                server.model, global_data, config, 1, target='valid', full_rank=True
            )
            current_valid_ndcg10 = valid_results.get('NDCG10', 0)
            if current_valid_ndcg10 > best_valid_ndcg10:
                best_valid_ndcg10 = current_valid_ndcg10
                selected_round = r + 1
                selected_results = dict(avg_results)
                selected_std = dict(std_results)
                selected_valid = dict(valid_results)
                best_model_state = deepcopy(server.model.state_dict())
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            # Track the best personalized model per client. Only the clients that
            # were actually updated this round can have changed, so we skip the rest.
            for client in selected_clients:
                c_valid, _ = test_server(
                    client.model, global_data, config, 1, target='valid', full_rank=True
                )
                c_ndcg10 = c_valid.get('NDCG10', 0)
                if c_ndcg10 > best_client_valid[client.id]:
                    best_client_valid[client.id] = c_ndcg10
                    best_client_state[client.id] = deepcopy(client.model.state_dict())
                    best_client_round[client.id] = r + 1

            # Update progress bar to display current performance
            current_ndcg10 = avg_results.get('NDCG10', 0)
            current_hr10 = avg_results.get('HR10', 0)
            progress_bar.set_postfix({
                'NDCG@10': f'{current_ndcg10:.4f}',
                'HR@10': f'{current_hr10:.4f}',
                'Val_NDCG@10': f'{current_valid_ndcg10:.4f}',
                'Best_Val@10': f'{best_valid_ndcg10:.4f}',
                'Best_round': selected_round
            })

            # Compare current server performance with best performance and update if better
            for metric in avg_results.keys():
                if avg_results[metric] > max_results_dict['mean'][metric]:
                    max_results_dict['mean'][metric] = avg_results[metric]
                    max_results_dict['std'][metric] = std_results[metric]

            # Early stopping based on the validation signal
            if early_stop_patience and rounds_without_improvement >= early_stop_patience:
                logging.info(
                    f"Early stopping at round {r + 1}: no validation improvement "
                    f"for {rounds_without_improvement} rounds (best round {selected_round})."
                )
                print(
                    f"\n⏹️  Early stopping at round {r + 1} "
                    f"(best validation round: {selected_round})"
                )
                break
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
    finally:
        # Ensure progress bar is closed in any case
        if 'progress_bar' in locals():
            progress_bar.close()

    # Display training completion summary (validation-selected test performance)
    best_ndcg10 = selected_results.get('NDCG10', 0)
    best_hr10 = selected_results.get('HR10', 0)

    print(f"\n✅ Training completed!")
    print(f"🏆 Test Performance @ best validation round ({selected_round}) - NDCG@10: {best_ndcg10:.4f}, HR@10: {best_hr10:.4f}")
    print(f"📊 Algorithm: {config['algorithm']}, Dataset: {config['dataset'].upper()}")
    print(f"⏱️  Total rounds: {config['rounds']}")

    # Write maximum performance and corresponding standard deviation to txt file
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    main_path = f"output/Results_{config['data_mode']}_{current_time}/{config['dataset']}/{config['algorithm']}_{config['num_clients']}"
    txt_path = main_path + f"/server/best_performance.txt"
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    # Primary result: test metrics at the round with the best validation NDCG@10
    with open(txt_path, 'w') as txtfile:
        txtfile.write(f"# selected by best validation NDCG@10 (full-rank over all POIs) at round {selected_round}\n")
        txtfile.write(f"# validation NDCG@10 at selected round: {round(best_valid_ndcg10, 2)}\n")
        for metric in selected_results:
            txtfile.write(f"{metric}: {round(selected_results[metric], 2)}, Std: {round(selected_std[metric], 2)}\n")

    # Reference only: optimistic per-metric maximum over all rounds on the test set
    optimistic_path = main_path + f"/server/best_performance_optimistic_max.txt"
    with open(optimistic_path, 'w') as txtfile:
        for metric in max_results_dict['mean']:
            max_results_dict['mean'][metric] = round(max_results_dict['mean'][metric], 2)
            max_results_dict['std'][metric] = round( max_results_dict['std'][metric], 2)
            txtfile.write(f"{metric}: {max_results_dict['mean'][metric]}, Std: {max_results_dict['std'][metric]}\n")

    # Save server performance for each round
    path = main_path + f"/server/all_rounds_results.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        #print(server_results)
        if server_results:  # Check if server_results is empty
            metrics = list(server_results[0].keys())
            writer.writerow(['rounds'] + metrics)  # Write header
            for i, results in enumerate(server_results):
                row = [i + 1] + [results[metric] for metric in metrics]
                writer.writerow(row)
        else:
            # If no results, write empty header
            writer.writerow(['rounds', 'NDCG5', 'NDCG10', 'NDCG20', 'HR5', 'HR10', 'HR20'])

    path = main_path + f"/model_weight/server/"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(server.model, path + "last_round_server_model.pt")

    # Persist the validation-selected (best) global model. Restore its weights
    # first so that this saved model -- and the per-client final tests below --
    # use the round that validation actually selected, not the last round.
    server.model.load_state_dict(best_model_state)
    torch.save(server.model, path + "best_server_model.pt")


    t2 = time.perf_counter()
    all_time = t2 - t1

    # Save training time
    path = main_path + f"/"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_time_infors(path, all_time)
    save_memory_infors(path, server_send_size, server_recive_size)


    for c in range(config['num_clients']):
        client_model = clients[c].model

        # Save the last-round personalized model for reference, then restore the
        # validation-selected best weights for the reported test + saved model.
        last_client_path = main_path + f"/model_weight/client/client{c}_last_round_model.pt"
        os.makedirs(os.path.dirname(last_client_path), exist_ok=True)
        torch.save(client_model, last_client_path)

        client_model.load_state_dict(best_client_state[c])
        logging.info(
            f"client[{c}] best personalized model selected at round "
            f"{best_client_round[c]} (val NDCG@10={best_client_valid[c]:.4f})"
        )

        path = main_path + f"/final_test/client/client[{c}]_results.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        test_model_final(client_model, global_data, path, config, time=3, c_id = c)

        # Persist each client's best personalized model so it can be used for
        # inference at runtime (e.g. in inference.ipynb).
        client_model_path = main_path + f"/model_weight/client/client{c}_model.pt"
        os.makedirs(os.path.dirname(client_model_path), exist_ok=True)
        torch.save(client_model, client_model_path)

    logging.info("Results has saved!")

