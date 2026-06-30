"""
predict.py - Generate next-POI predictions with a trained FedKG model.

Each trained model is a SASRec sequence recommender that takes a user's recent
sequence of check-in POI IDs and scores every POI in the catalog as a candidate
for the *next* check-in. This script loads either the federated SERVER model or
one of the per-client (personalized) CLIENT models and ranks the most likely
next POIs for a given check-in sequence.

Examples
--------
# Use the global server model on an inline sequence of POI IDs:
python predict.py --model server --checkins "1,593,617,5" --topk 10

# Use the personalized model of client 2, reading the sequence from a file
# (one POI ID per line, or a single comma/space separated line):
python predict.py --model client --client-id 2 --checkins-file seq.txt

# Point directly at a saved .pt file:
python predict.py --model-path output/.../model_weight/server/best_server_model.pt \
                  --checkins "1,593,617"
"""

import argparse
import glob
import os

import numpy as np
import torch

# Importing the model class is required so that torch.load can un-pickle the
# saved SASRec objects (they were stored with `torch.save(model, path)`).
from src.models import SASRec  # noqa: F401


def find_latest_run_dir():
    """Return the most recent output/Results_* run directory (or None)."""
    candidates = glob.glob("output/Results_*/*/*")  # output/Results_<ts>/<dataset>/<algo>_<n>
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def resolve_model_path(args):
    """Turn the high-level --model selection into a concrete .pt file path."""
    # An explicit path always wins.
    if args.model_path:
        return args.model_path

    run_dir = args.run_dir or find_latest_run_dir()
    if run_dir is None:
        raise FileNotFoundError(
            "No trained model found under ./output. Train a model first or pass "
            "--model-path to point at a saved .pt file."
        )

    if args.model == "server":
        # best_server_model.pt is the validation-selected global model.
        path = os.path.join(run_dir, "model_weight", "server", "best_server_model.pt")
    else:  # client
        if args.client_id is None:
            raise ValueError("--client-id is required when --model client is selected.")
        # client<i>_model.pt is the validation-selected personalized client model.
        path = os.path.join(
            run_dir, "model_weight", "client", f"client{args.client_id}_model.pt"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected model file does not exist: {path}")
    return path


def load_model(model_path, device):
    """Load a full SASRec model object and move it onto the requested device."""
    # weights_only=False because we saved the whole module, not just a state_dict.
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.dev = device          # the stored model may still point at the training device (e.g. cuda:0)
    model.to(device)
    model.eval()
    return model


def parse_checkins(args):
    """Read the input check-in sequence (list of POI IDs) from CLI or file."""
    if args.checkins_file:
        with open(args.checkins_file, "r", encoding="ISO-8859-1") as f:
            raw = f.read()
    else:
        raw = args.checkins
    # Accept commas, whitespace or newlines as separators.
    tokens = raw.replace(",", " ").split()
    seq = [int(t) for t in tokens]
    if not seq:
        raise ValueError("Empty check-in sequence.")
    return seq


def load_poi_names(dataset):
    """Optional: map POI ID -> venue category name for readable output."""
    path = f"data/processed_{dataset}.txt"
    names = {}
    if not os.path.exists(path):
        return names
    with open(path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) >= 4:
                names.setdefault(int(parts[1]), parts[3])  # col 2 = POI ID, col 4 = category name
    return names


def predict_next(model, checkins, maxlen, device, topk):
    """Rank all POIs as candidates for the next check-in after `checkins`."""
    # Build the left-padded fixed-length sequence the model expects: the most
    # recent check-in sits at the last position, older ones precede it, and any
    # leading slots are 0 (the padding index). This mirrors the eval pipeline.
    seq = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1
    for poi in reversed(checkins):
        if idx < 0:
            break
        seq[idx] = poi
        idx -= 1

    # Score the sequence against the entire POI catalog (full ranking).
    item_num = model.item_num
    all_items = np.arange(1, item_num + 1)
    with torch.no_grad():
        # predict() returns logits of shape (1, item_num); higher = more likely.
        logits = model.predict(np.array([0]), np.array([seq]), all_items)[0]
    scores = logits.detach().cpu().numpy()

    # Top-k POI IDs by score (POI id = score index + 1).
    top_idx = np.argsort(-scores)[:topk]
    return [(int(i + 1), float(scores[i])) for i in top_idx]


def main():
    parser = argparse.ArgumentParser(description="Generate next-POI predictions from a trained FedKG model.")

    # --- Which model to use ---
    parser.add_argument("--model", choices=["server", "client"], default="server",
                        help="Use the global server model or a per-client personalized model.")
    parser.add_argument("--client-id", type=int, default=None,
                        help="Client index to load when --model client is used.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Direct path to a saved .pt model (overrides --model/--client-id).")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Results run directory to load from (default: most recent under ./output).")

    # --- Input check-in sequence ---
    parser.add_argument("--checkins", type=str, default=None,
                        help="Check-in sequence as comma/space separated POI IDs, e.g. \"1,593,617\".")
    parser.add_argument("--checkins-file", type=str, default=None,
                        help="File containing the check-in POI IDs (whitespace/comma/newline separated).")

    # --- Output / runtime ---
    parser.add_argument("--topk", type=int, default=10, help="Number of next-POI candidates to return.")
    parser.add_argument("--maxlen", type=int, default=None,
                        help="Max sequence length (default: inferred from the model).")
    parser.add_argument("--dataset", type=str, default="nyc",
                        help="Dataset name, used only to look up POI category names for display.")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device, e.g. cpu or cuda:0 (default: cuda if available, else cpu).")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = resolve_model_path(args)
    model = load_model(model_path, device)
    # The positional embedding table size equals the training maxlen.
    maxlen = args.maxlen or model.pos_emb.num_embeddings

    checkins = parse_checkins(args)
    poi_names = load_poi_names(args.dataset)

    results = predict_next(model, checkins, maxlen, device, args.topk)

    # --- Report ---
    print(f"Model:    {model_path}")
    print(f"Device:   {device}")
    print(f"Input check-in sequence ({len(checkins)} POIs): {checkins}")
    print(f"\nTop-{args.topk} predicted next POIs:")
    print(f"{'rank':>4}  {'POI_id':>7}  {'score':>10}  category")
    for rank, (poi_id, score) in enumerate(results, start=1):
        name = poi_names.get(poi_id, "")
        print(f"{rank:>4}  {poi_id:>7}  {score:>10.4f}  {name}")


if __name__ == "__main__":
    main()
