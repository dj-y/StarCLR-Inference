import os
import json
import argparse
import multiprocessing as mp
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file

import pandas as pd
from tqdm import tqdm

from BertContrastiveLearningModel import (
    MLPModelTESS,
    MLPModelZTF,
    MLPModelGaia,
)
import Preprocess
from DataLoading import RepresentationDataset, collate_representations


# ============================================================
# Stage-2 Inference Script (Downstream Head Only)
#   - Input: exported representations from Stage-1 (hidden_states)
#   - Model: lightweight MLP head
#   - Output: predicted label + probability
# ============================================================


def main():
    # =================== Command-line arguments ===================
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["tess", "ztf", "gaia"],
        help="Target dataset used for stage-2 prediction"
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for stage-2 inference"
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cpu, cuda, cuda:0"
    )
    args = ap.parse_args()


    # =================== Dataset-specific model selection ===================
    if args.dataset == "tess":
        ModelForSequenceClassification = MLPModelTESS
    elif args.dataset == "ztf":
        ModelForSequenceClassification = MLPModelZTF
    elif args.dataset == "gaia":
        ModelForSequenceClassification = MLPModelGaia
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


    # =================== Paths and parameters ===================
    # NOTE: Stage-2 consumes Stage-1 exported representations.
    file = f"output/hidden_states_{args.dataset}.parquet"  # change to .csv if Stage-1 saved csv
    model_path = f"checkpoints/{args.dataset}"

    batch_size = args.batch_size
    num_workers = max(1, (os.cpu_count() or 1) // 4)

    output_file = f"output/predictions_label_{args.dataset}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    # =================== Read exported representations ===================
    print("Reading extracted representations...")
    df = pd.read_parquet(file)


    # =================== Parallel preprocessing ===================
    # Convert rows into dicts compatible with RepresentationDataset/collate_representations.
    print("Starting parallel preprocessing for hidden states...")
    with mp.Pool(processes=num_workers) as pool:
        preprocessed = list(
            tqdm(
                pool.imap(
                    Preprocess.preprocess_row_hidden_states,
                    [row for _, row in df.iterrows()],
                ),
                total=len(df),
            )
        )


    # =================== Load backbone config.json and inject as model.config ===================
    backbone_config_path = os.path.join(model_path, "config.json")
    with open(backbone_config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config_dict["id2label"] = {int(k): v for k, v in config_dict["id2label"].items()}
    config_dict["label2id"] = {k: int(v) for k, v in config_dict["label2id"].items()}
    config_dict["num_labels"] = len(config_dict["id2label"])

    # Build a lightweight config object
    config = SimpleNamespace(**config_dict)


    # =================== Load Stage-2 model (MLP head) ===================
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested, but CUDA is not available.")
        if (
            device.type == "cuda"
            and device.index is not None
            and device.index >= torch.cuda.device_count()
        ):
            raise RuntimeError(
                f"Requested device index {device.index} is out of range. "
                f"Available CUDA devices: {torch.cuda.device_count()}."
            )
    print(f"Using device: {device}")

    model = ModelForSequenceClassification(config)

    state_dict_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(state_dict_path)
    model.load_state_dict(state_dict, strict=True)

    # Wrap DP
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        model.module.config = config
    else:
        model.config = config

    model.to(device)
    model.eval()

    # Use HF-like access pattern
    config = model.module.config if hasattr(model, "module") else model.config
    id2label = config.id2label
    label2id = config.label2id


    # =================== Inference ===================
    dataloader = DataLoader(
        RepresentationDataset(preprocessed),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_representations,
        num_workers=num_workers,
    )

    results = []
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running stage-2 inference (MLP head)"):
            hidden_states = batch["hidden_states"].to(device)
            feature = batch["feature"].to(device)

            outputs = model(hidden_states=hidden_states, feature=feature)
            probs = F.softmax(outputs.logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()

            for i, meta in enumerate(batch["meta"]):
                true_type = meta.get("Type", None)

                pred_id = int(preds[i])
                pred_label = id2label[pred_id]
                pred_prob = float(probs[i, pred_id].item())

                results.append(
                    {
                        "Type": true_type,
                        "predicted_label": pred_label,
                        "predicted_probability": pred_prob,
                    }
                )

                # Optional: keep for evaluation if GT is available and in label2id
                if true_type is not None and true_type in label2id:
                    y_true.append(int(label2id[true_type]))
                    y_pred.append(pred_id)


    # =================== Save results ===================
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    print(f"Stage-2 inference completed. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
