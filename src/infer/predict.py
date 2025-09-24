import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import argparse
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from BertContrastiveLearningModel import (
    BertForSequenceClassificationTESS, 
    BertForSequenceClassificationZTF, 
    BertForSequenceClassificationGaia
)
import Preprocess
from DataLoading import PreprocessedDataset, collate_fn

# =================== Command-line arguments ===================
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["tess", "ztf", "gaia"])
ap.add_argument("--batch_size", type=int, default=16)
args = ap.parse_args()

# Select dataset-specific preprocessing function and model
if args.dataset == "tess":
    preprocess_row = Preprocess.preprocess_row_tess
    BertForSequenceClassification = BertForSequenceClassificationTESS
    flux = 'Flux'
elif args.dataset == "ztf":
    preprocess_row = Preprocess.preprocess_row_ztf
    BertForSequenceClassification = BertForSequenceClassificationZTF
    flux = 'mag'
elif args.dataset == "gaia":
    preprocess_row = Preprocess.preprocess_row_gaia
    BertForSequenceClassification = BertForSequenceClassificationGaia
    flux = 'g_transit_flux'
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")


# =================== Paths and parameters ===================
file = f'example/example_{args.dataset}.parquet'
model_path = f'checkpoints/{args.dataset}'
batch_size = args.batch_size
num_workers = os.cpu_count() // 4
output_file = f'output/predictions_label_{args.dataset}.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)


# =================== Data preparation (sorting + parallel preprocessing) ===================
print("Reading data and sorting...")
df = pd.read_parquet(file)
df['length'] = df[flux].apply(len)
df = df.sort_values('length').reset_index(drop=True)

print("Starting parallel preprocessing...")
with mp.Pool(processes=num_workers) as pool:
    preprocessed = list(tqdm(pool.imap(preprocess_row, [row for _, row in df.iterrows()]), total=len(df)))


# =================== Load model ===================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(model_path)
model = torch.nn.DataParallel(model)
model.to(device)
model.eval()

config = model.module.config
id2label = config.id2label

# =================== Inference ===================
dataloader = DataLoader(PreprocessedDataset(preprocessed), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=num_workers)

results = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Running inference"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        feature = batch['feature'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Collect minimal results
        for i, meta in enumerate(batch['meta']):
            results.append({
                'Type': meta['Type'],
                'predicted_label': id2label[preds[i].item()],
                'predicted_probability': probs[i, preds[i]].item()
            })


# =================== Save results ===================
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

print(f"Inference completed. Results saved to: {output_file}")
