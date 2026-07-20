
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict


class SCTTNetDataset(Dataset):

    def __init__(
        self,
        X_meteo_list: List[Dict],
        X_gate_list: List[np.ndarray],
        y_list: List[np.ndarray]
    ):
        self.X_meteo_list = X_meteo_list
        self.X_gate_list = X_gate_list
        self.y_list = y_list

        assert len(X_meteo_list) == len(X_gate_list) == len(y_list), \
            "All input lists must have the same length"

    def __len__(self):
        return len(self.X_meteo_list)

    def __getitem__(self, idx):
        meteo_input = {
            'enc_x': torch.FloatTensor(self.X_meteo_list[idx]['enc_x']),
            'dec_x': torch.FloatTensor(self.X_meteo_list[idx]['dec_x'])
        }
        gate_input = torch.FloatTensor(self.X_gate_list[idx])
        target = torch.FloatTensor(self.y_list[idx])

        return meteo_input, gate_input, target


def collate_fn(batch):
    meteo_inputs, gate_inputs, targets = zip(*batch)

    enc_x_batch = torch.stack([m['enc_x'] for m in meteo_inputs])
    dec_x_batch = torch.stack([m['dec_x'] for m in meteo_inputs])

    batched_meteo = {
        'enc_x': enc_x_batch,
        'dec_x': dec_x_batch
    }

    batched_gate = torch.stack(gate_inputs)
    batched_target = torch.stack(targets)

    return batched_meteo, batched_gate, batched_target


def prepare_dataloaders(
    train_data: Tuple,
    val_data: Tuple,
    test_data: Tuple,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    print("\nPreparing PyTorch DataLoaders...")

    train_X_meteo, train_X_gate, train_y = train_data
    train_dataset = SCTTNetDataset(train_X_meteo, train_X_gate, train_y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")

    val_X_meteo, val_X_gate, val_y = val_data
    val_dataset = SCTTNetDataset(val_X_meteo, val_X_gate, val_y)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"Validation: {len(val_dataset)} samples, {len(val_loader)} batches")

    test_X_meteo, test_X_gate, test_y = test_data
    test_dataset = SCTTNetDataset(test_X_meteo, test_X_gate, test_y)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    print("=" * 60)
    print("Testing SCTTNetDataset")
    print("=" * 60)

    n_samples = 100
    seq_len = 96
    label_len = 48
    pred_len = 1
    n_meteo_features = 8
    n_gate_features = 3

    X_meteo_list = []
    X_gate_list = []
    y_list = []

    for i in range(n_samples):
        meteo_input = {
            'enc_x': np.random.randn(seq_len, n_meteo_features),
            'dec_x': np.random.randn(label_len + pred_len, n_meteo_features)
        }
        gate_input = np.random.randn(seq_len, n_gate_features)
        target = np.random.randn(pred_len, 1)

        X_meteo_list.append(meteo_input)
        X_gate_list.append(gate_input)
        y_list.append(target)

    print("\nTest 1: Create SCTTNetDataset")
    dataset = SCTTNetDataset(X_meteo_list, X_gate_list, y_list)
    print(f"Dataset samples: {len(dataset)}")

    print("\nTest 2: Get single sample")
    meteo, gate, target = dataset[0]
    print(f"meteo['enc_x'] shape: {meteo['enc_x'].shape}")
    print(f"meteo['dec_x'] shape: {meteo['dec_x'].shape}")
    print(f"gate shape: {gate.shape}")
    print(f"target shape: {target.shape}")

    print("\nTest 3: Create DataLoader")
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )
    print(f"DataLoader batches: {len(dataloader)}")

    print("\nTest 4: Iterate batches")
    for batch_idx, (meteo_batch, gate_batch, target_batch) in enumerate(dataloader):
        if batch_idx < 2:
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  meteo['enc_x'] shape: {meteo_batch['enc_x'].shape}")
            print(f"  meteo['dec_x'] shape: {meteo_batch['dec_x'].shape}")
            print(f"  gate shape: {gate_batch.shape}")
            print(f"  target shape: {target_batch.shape}")

    print("\nTest 5: prepare_dataloaders")
    train_data = (X_meteo_list[:70], X_gate_list[:70], y_list[:70])
    val_data = (X_meteo_list[70:85], X_gate_list[70:85], y_list[70:85])
    test_data = (X_meteo_list[85:], X_gate_list[85:], y_list[85:])

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_data, val_data, test_data,
        batch_size=16,
        shuffle_train=True
    )

    print(f"\nTrain DataLoader batches: {len(train_loader)}")
    print(f"Validation DataLoader batches: {len(val_loader)}")
    print(f"Test DataLoader batches: {len(test_loader)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
