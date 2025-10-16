from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from ..datasets.face_window_dataset import RandomFaceWindowDataset
from ..models import Combined


def masked_mse_loss(predictions: Tensor, targets: Tensor, visibility: Tensor) -> Tensor:
    """Compute a visibility-weighted mean squared error."""

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must share the same shape")

    weights = visibility.to(dtype=predictions.dtype)
    if weights.shape != predictions.shape:
        raise ValueError("visibility must align with predictions for masking")

    squared_error = (predictions - targets) ** 2
    clamped_weights = weights.clamp_min(0.0)
    weighted_error = squared_error * clamped_weights
    normalizer = clamped_weights.sum().clamp_min(torch.finfo(predictions.dtype).eps)
    return weighted_error.sum() / normalizer


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_dataloader(
    dataset: RandomFaceWindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: Optional[bool],
) -> DataLoader:
    use_pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )


def train_one_epoch(
    model: Combined,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    batches = 0

    for batch in dataloader:
        frames = batch["frames"].to(device)
        metadata = batch["face_metadata"].to(device)
        targets = batch["heart_rates"].to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions, _, visibility = model(frames, metadata)
        loss = loss_fn(predictions, targets, visibility.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches += 1

    return running_loss / max(batches, 1)


def evaluate(
    model: Combined,
    dataloader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device)
            metadata = batch["face_metadata"].to(device)
            targets = batch["heart_rates"].to(device)

            predictions, _, visibility = model(frames, metadata)
            loss = loss_fn(predictions, targets, visibility.to(device))

            running_loss += loss.item()
            batches += 1

    return running_loss / max(batches, 1)


def train(
    *,
    model: Optional[Combined] = None,
    train_dataset: Optional[RandomFaceWindowDataset] = None,
    val_dataset: Optional[RandomFaceWindowDataset] = None,
    train_dataset_kwargs: Optional[Dict[str, object]] = None,
    val_dataset_kwargs: Optional[Dict[str, object]] = None,
    batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    loss_fn: Optional[Callable[[Tensor, Tensor, Tensor], Tensor]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, list]:
    """Train a combined model using :class:`RandomFaceWindowDataset` samples."""

    if train_dataset is None:
        train_dataset = RandomFaceWindowDataset(**(train_dataset_kwargs or {}))
    if val_dataset is None and val_dataset_kwargs is not None:
        val_dataset = RandomFaceWindowDataset(**val_dataset_kwargs)

    resolved_device = _resolve_device(device)

    model = model or Combined()
    model.to(resolved_device)

    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = loss_fn or masked_mse_loss

    train_loader = _make_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = _make_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

    history: Dict[str, list] = {"train_loss": []}
    if val_loader is not None:
        history["val_loss"] = []

    for _ in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, resolved_device)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, loss_fn, resolved_device)
            history["val_loss"].append(val_loss)

        if scheduler is not None:
            scheduler.step()

    return history


__all__ = [
    "train",
    "train_one_epoch",
    "evaluate",
    "masked_mse_loss",
]
