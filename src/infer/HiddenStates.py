import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import argparse
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from BertContrastiveLearningModel import BertForPositionEmbeddingHiddenStates
import Preprocess
from DataLoading import ModelInputDataset, collate_model_inputs


# ============================================================
# This script performs **Stage-1 inference** for StarCLR:
#   - It does NOT perform final classification or evaluation
#   - It only extracts and saves model representations
#
# The exported representations (hidden states / embeddings)
# are intended for:
#   - downstream fine-tuning
#   - linear probing
#   - UMAP / t-SNE visualization
#   - other representation analysis
#
# This design makes the pipeline modular and reusable.
# ============================================================


def main():
    # =================== Command-line arguments ===================
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["tess", "ztf", "gaia"],
        help="Target dataset used for representation extraction"
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for representation extraction"
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cpu, cuda, cuda:0"
    )
    args = ap.parse_args()

    # =================== Dataset-specific preprocessing ===================
    # Select preprocessing function and flux field according to dataset.
    # This keeps the representation-extraction logic dataset-agnostic.
    if args.dataset == "tess":
        preprocess_row = Preprocess.preprocess_row_tess
        flux = "Flux"
    elif args.dataset == "ztf":
        preprocess_row = Preprocess.preprocess_row_ztf
        flux = "mag"
    elif args.dataset == "gaia":
        preprocess_row = Preprocess.preprocess_row_gaia
        flux = "g_transit_flux"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # =================== Paths and basic parameters ===================
    # Input light-curve file
    file = f"example/example_{args.dataset}.parquet"

    batch_size = args.batch_size
    num_workers = max(1, (os.cpu_count() or 1) // 4)

    # Output file storing extracted representations
    output_file = f"output/hidden_states_{args.dataset}.parquet"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # =================== Data preparation ===================
    # Step 1:
    #   - Read parquet file
    #   - Sort samples by sequence length to reduce padding overhead
    print("Reading data and sorting by sequence length...")
    df = pd.read_parquet(file)
    df["length"] = df[flux].apply(len)
    df = df.sort_values("length").reset_index(drop=True)

    # Step 2:
    #   - Parallel preprocessing
    #   - Convert raw light curves into model-ready inputs
    print("Starting parallel preprocessing...")
    with mp.Pool(processes=num_workers) as pool:
        preprocessed = list(
            tqdm(
                pool.imap(preprocess_row, [row for _, row in df.iterrows()]),
                total=len(df)
            )
        )

    # =================== Load pretrained model ===================
    # Load StarCLR backbone with pretrained weights.
    # The model is used purely as a feature extractor here.
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

    model = BertForPositionEmbeddingHiddenStates.from_pretrained("checkpoints/backbone")

    # Wrap DP
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        config = model.module.config
    else:
        config = model.config
    id2label = config.id2label

    model.to(device)
    model.eval()

    # =================== DataLoader ===================
    dataloader = DataLoader(
        ModelInputDataset(preprocessed),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_model_inputs,
        num_workers=num_workers,
    )

    # =================== Stage-1 Inference: Representation Extraction ===================
    # This loop performs forward passes only.
    # No loss computation, no backpropagation.
    # The outputs are intermediate representations rather than final predictions.
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting hidden representations"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            feature = batch["feature"].to(device)

            # Forward pass
            # The model is configured to expose hidden representations
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                feature=feature,
            )

            # Here, `logits` are treated as fixed-length representations
            # rather than task-specific predictions.
            hidden_states_cpu = outputs.logits.cpu().numpy()
            feature_cpu = batch["feature"].cpu().numpy()
            metas = batch["meta"]

            # Store representations together with minimal metadata
            for i in range(len(metas)):
                meta = metas[i]
                results.append({
                    "hidden_states": hidden_states_cpu[i],
                    "feature": feature_cpu[i],
                    "Type": meta["Type"],   # label kept for downstream analysis
                })

    # =================== Save extracted representations ===================
    # The saved file can be reused for:
    #   - downstream fine-tuning
    #   - linear probing
    #   - UMAP visualization
    #   - cross-survey representation analysis
    results_df = pd.DataFrame(results)
    results_df.to_parquet(output_file, index=False)

    print(f"Stage-1 inference completed. Representations saved to: {output_file}")


if __name__ == "__main__":
    main()
