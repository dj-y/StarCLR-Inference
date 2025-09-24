import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'feature': torch.tensor(item['feature']),
            'meta': item['meta']
        }


def collate_fn(batch):
    max_len = max(item['input_ids'].shape[0] for item in batch)
    input_ids = []
    attention_masks = []
    features = []
    metas = []

    for item in batch:
        pad_len = max_len - item['input_ids'].shape[0]
        input_ids.append(torch.cat([item['input_ids'], torch.zeros((pad_len, 2))]))
        attention_masks.append(torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.int8)]))
        features.append(item['feature'])
        metas.append(item['meta'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'feature': torch.stack(features),
        'meta': metas
    }