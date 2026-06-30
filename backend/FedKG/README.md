# FedKG — Federated POI Recommendation for Spatial Heterogeneity

FedKG is a **federated learning** framework for **next-POI (Point-of-Interest) recommendation**.
Several clients each hold the check-in trajectories of users from *different geographic
regions* and train a shared sequence model **without ever sharing the raw trajectories**.

The framework combines two ideas to deal with *spatial heterogeneity* (some clients have
dense city data, others sparse suburban data):

1. **Bidirectional Knowledge Distillation** — clients and server teach each other so the
   global model keeps personalized local knowledge *and* shared global knowledge.
2. **ST-GAN data augmentation (optional)** — a Spatio-Temporal GAN synthesises extra
   trajectories to "repair" fragmented spatial patterns. **This module is optional: the
   framework trains with or without it (see [Running with and without the GAN](#running-with-and-without-the-gan)).**

The base recommender is **SASRec** (a self-attention sequence model). Performance is
reported with standard ranking metrics (NDCG@K, HR@K).

---

## Table of Contents
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Running with and without the GAN](#running-with-and-without-the-gan)
- [Configuration reference](#configuration-reference)
- [Data format](#data-format)
- [Outputs](#outputs)
- [Helper scripts](#helper-scripts)

---

## How it works

One *communication round* of training looks like this (see `main.py`):

1. **Server → clients (Personalized KD):** the server sends the global model to each
   client. The client distills it against its own previous model so the update keeps
   personalized knowledge (`FedKGClient.kd_train`).
2. **Local training:** each selected client trains the model on its local trajectories
   (`FedKGClient.client_update`).
3. **Clients → server (Global KD):** the server receives each client model and distills
   it against the current global model on a small global dataset
   (`FedKGServer.norm_train`).
4. **Aggregation:** the server averages the client models (weighted by data size) into the
   new global model (`FedKGServer.aggregate`).

After every round the server and client models are evaluated, and the round with the best
**validation NDCG@10** is selected and saved (validation-based model selection — not the
optimistic per-metric maximum over the test set).

The **optional ST-GAN stage** runs *once, before* the rounds begin: it pretrains a GAN per
client, aggregates a global generator, samples synthetic trajectories, and merges them with
the real data. When the GAN is disabled, training simply uses the original data instead —
**the training loop above is identical either way.**

---

## Project structure

```
FedKG/
├── main.py                     # Entry point: builds models, runs the federated training loop
├── configs_tuned.json          # Configuration (JSON with // comments)
├── requirements.txt            # Python dependencies
├── predict.py                  # Rank next-POI predictions with a trained model
├── plot_training_progress.py   # Plot per-round test metrics from a finished run
├── preprocessing.py            # Clean/filter the raw Foursquare check-in data
├── split_data.ipynb            # Split a dataset into per-client federated partitions
│
├── data/
│   ├── processed_nyc.txt       # Full processed NYC dataset (used as the "global" data)
│   ├── dataset_TSMC2014_*.txt  # Raw Foursquare check-in datasets (NYC, TKY)
│   └── space/nyc/              # "space" partition for the NYC dataset
│       ├── client{0..4}_data.txt        # Per-client original trajectories
│       ├── merged_client{0..4}_data.txt # Original + GAN-synthesised data (only used if GAN ran)
│       └── aug_server_data.txt          # GAN-synthesised global data for server-side KD
│
├── gan/                        # GAN checkpoints (created when the GAN stage runs)
│   └── {mode}/{dataset}/...    #   gen_client*.pth, dis_client*.pth, global_generator_model.pth
│
├── logs/                       # Training logs
├── output/                     # Results of each run (metrics, models, timings) — see Outputs
│
└── src/
    ├── util.py                 # Data loading, sampling, evaluation (NDCG/HR)
    ├── models.py               # SASRec model definition
    ├── fedoptimizer.py         # Federated optimizer helpers
    └── algorithms/
        ├── FedKG/
        │   ├── server.py       # FedKGServer: aggregation + global knowledge distillation
        │   └── client.py       # FedKGClient: local training + personalized knowledge distillation
        └── STGAN/              # Optional ST-GAN data-augmentation module
            ├── gan_trainer.py  # GAN training + synthetic-trajectory generation
            ├── gan_models.py   # Generator / Discriminator
            ├── data_setting.py # Prepares GAN input from the processed dataset
            └── helpers.py      # GAN batching/loss helpers
```

> The currently shipped split is `data_mode = "space"` for the `nyc` dataset with 5 clients.
> Use `split_data.ipynb` to create splits for other datasets/client counts.

---

## Installation

Requires **Python 3.7+**. A GPU is optional (set `"device": "cpu"` to run on CPU).

```bash
pip install -r requirements.txt
```

Dependencies: PyTorch, NumPy, Pandas, tqdm, scikit-learn.
`plot_training_progress.py` additionally needs `matplotlib`.

---

## Quick start

```bash
python main.py --config_path configs_tuned.json
```

`configs_tuned.json` is the default, tuned configuration. It is plain JSON but allows
`//` line comments, which `main.py` strips before parsing — so you can document your
settings inline. The JSON values are authoritative for the model hyperparameters (the
argparse defaults in `main.py` are only fallbacks).

When training finishes, results are written under `output/Results_<mode>_<timestamp>/...`
(see [Outputs](#outputs)).

---

## Running with and without the GAN

The ST-GAN augmentation module is **fully optional** and controlled by two flags in the
config. The federated **training logic is identical** in both modes — only the *input data*
changes.

| Goal | `pretrain_gan` | `generate_aug_data` | `use_original_data` | Data used for training |
|------|----------------|---------------------|---------------------|------------------------|
| **Without GAN** (default) | `"False"` | `"False"` | `"False"` | Original client data; if pre-generated `merged_*`/`aug_server` files exist they are reused, otherwise it falls back to the raw client data automatically. |
| **True no-GAN baseline** | `"False"` | `"False"` | `"True"` | Always the raw `client{id}_data.txt`, ignoring any pre-generated augmented files. |
| **With GAN** | `"True"` | `"True"` | `"False"` | Trains the GANs, generates synthetic trajectories, then trains on the merged (original + synthetic) data. |

Notes:
- When both GAN flags are `"False"`, the GAN module is **never even constructed** — there is
  no dependency on the GAN code or checkpoints.
- If a `merged_client*_data.txt` (client) or `aug_server_data.txt` (server) file is missing,
  the code transparently falls back to the original `client{id}_data.txt` / the full
  `processed_{dataset}.txt`, so a run never crashes for a missing augmented file.

---

## Configuration reference

Every key is documented inline in `configs_tuned.json`. The most important ones:

#### Federated setup
- `algorithm` — federated algorithm to run (`"FedKG"`).
- `dataset` — dataset name (`"nyc"`).
- `data_mode` — partition folder under `data/<mode>/<dataset>` (`"space"`).
- `num_clients` — number of federated clients.
- `rounds` — number of communication rounds.
- `jr` — join ratio: fraction of clients trained each round.
- `seed` — random seed.
- `early_stop_patience` — stop after this many rounds without validation improvement (`0` disables).

#### Model (SASRec)
- `model_name` — `"SASRec"`.
- `hidden_units`, `maxlen`, `num_heads`, `num_blocks`, `dropout_rate` — architecture.

#### Local training
- `client_epochs`, `client_batch_size`, `lr`, `l2_emb`, `weight_decay`.
- `use_lr_schedule`, `warmup_frac` — per-round warmup + cosine LR schedule.

#### Loss & evaluation
- `loss_type` — `"sampled_softmax"` or `"bce"`.
- `num_train_neg`, `num_neg` — negatives for training / sampled-rank evaluation.
- `eval_full_rank` — rank against the full POI catalog instead of sampled negatives.

#### Knowledge distillation
- `T` — softmax temperature.
- `kd_epochs`, `norm_epochs` — client- / server-side distillation epochs.
- `lamda`, `nor_lamda` — KD-vs-supervised loss mixing weights (client / server).

#### ST-GAN (optional)
- `pretrain_gan`, `generate_aug_data`, `use_original_data` — see
  [Running with and without the GAN](#running-with-and-without-the-gan).
- `local_n_users`, `global_n_users` — number of synthetic users to generate.
- `alpha` — GAN mixing weight.

---

## Data format

Each trajectory file is tab-separated with one check-in per line:

```
user_id  poi_id  category_id  category_name  latitude  longitude  timezone_offset  timestamp
```

Example (`data/space/nyc/client0_data.txt`):

```
17  3134  303  Neighborhood   40.7595  -73.8314  -240  2012-04-03T18:57:46Z
17  4485    7  Home (private) 40.7356  -73.8124  -240  2012-04-03T20:04:26Z
```

The model only consumes the `user_id` / `poi_id` sequence; the other columns provide the
spatio-temporal side information used by the optional GAN and by preprocessing.

- `data/processed_{dataset}.txt` — the full processed dataset (used for the global model
  and server-side distillation).
- `data/{mode}/{dataset}/client{id}_data.txt` — per-client partitions.
- `merged_client{id}_data.txt` / `aug_server_data.txt` — original **+** GAN-synthesised data
  (only present/used when the GAN stage has run).

---

## Outputs

Each run is written to a timestamped folder:

```
output/Results_{data_mode}_{timestamp}/{dataset}/{algorithm}_{num_clients}/
├── server/
│   ├── best_performance.txt                 # Test metrics at the best-validation round (primary result)
│   ├── best_performance_optimistic_max.txt  # Per-metric max over all rounds (reference only)
│   ├── all_rounds_results.csv               # Per-round test metrics
│   └── training_progress/                   # Optional progress CSV/PNG
├── final_test/client/client[*]_results.csv  # Per-client test metrics
├── model_weight/
│   ├── server/best_server_model.pt          # Validation-selected global model
│   ├── server/last_round_server_model.pt
│   └── client/client{c}_model.pt            # Per-client (personalized) best models
├── time.txt                                 # Training time
└── memory.txt                               # Communication cost (model bytes sent/received)
```

**Evaluation metrics:** `NDCG@K` and `HR@K` for `K = 5, 10, 20`.

---

## Helper scripts

- **`predict.py`** — rank the most likely next POIs for a check-in sequence using a trained
  server or client model:
  ```bash
  python predict.py --model server --checkins "1,593,617,5" --topk 10
  python predict.py --model client --client-id 2 --checkins-file seq.txt
  ```
- **`plot_training_progress.py`** — plot the per-round test metrics of a finished run:
  ```bash
  python plot_training_progress.py            # auto-detect the latest run
  python plot_training_progress.py --csv path/to/all_rounds_results.csv
  ```
- **`preprocessing.py`** — clean/filter the raw Foursquare check-in data (e.g. drop
  routine home/office-only users) before splitting.
- **`split_data.ipynb`** — split a processed dataset into per-client federated partitions.

---

## License

MIT License.
