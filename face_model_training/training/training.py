from __future__ import annotations

import math
from contextlib import nullcontext
import inspect
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.auto import tqdm
from torch.profiler import ProfilerActivity, profile as torch_profile

from ..datasets.face_window_dataset import RandomFaceWindowDataset
from ..models import Combined


_amp_module = getattr(torch, "amp", None)
if _amp_module is not None and hasattr(_amp_module, "GradScaler"):
    from torch.amp import GradScaler  # type: ignore[attr-defined]
    from torch.amp import autocast as _autocast  # type: ignore[attr-defined]

    _AMP_REQUIRES_DEVICE_TYPE = True
else:  # pragma: no cover - fallback for older PyTorch versions
    from torch.cuda.amp import GradScaler  # type: ignore
    from torch.cuda.amp import autocast as _autocast  # type: ignore

    _AMP_REQUIRES_DEVICE_TYPE = False


def masked_mse_loss(predictions: Tensor, targets: Tensor, visibility: Tensor) -> Tensor:
    """Compute a visibility-weighted mean squared error."""

    if predictions.shape == targets.shape and visibility.shape == predictions.shape:
        weights = visibility.to(dtype=predictions.dtype)

        squared_error = (predictions - targets) ** 2
        clamped_weights = weights.clamp_min(0.0)
        weighted_error = squared_error * clamped_weights
        normalizer = clamped_weights.sum().clamp_min(torch.finfo(predictions.dtype).eps)
        return weighted_error.sum() / normalizer

    # Support scalar predictions (B,) where visibility is [B, S].
    if predictions.dim() == targets.dim() == 1:
        pred = predictions.view(-1)
        tgt = targets.view(-1).to(dtype=pred.dtype, device=pred.device)

        if visibility.dim() == 2:
            weights = visibility.to(dtype=pred.dtype, device=pred.device).clamp_min(0.0).sum(dim=1)
        elif visibility.dim() == 1:
            weights = visibility.to(dtype=pred.dtype, device=pred.device).clamp_min(0.0)
        else:
            raise ValueError("visibility must be 1D or 2D for scalar predictions")

        if weights.shape[0] != pred.shape[0]:
            raise ValueError("visibility must align with batch size for scalar predictions")

        squared_error = (pred - tgt) ** 2
        normalizer = weights.sum().clamp_min(torch.finfo(pred.dtype).eps)
        return (squared_error * weights).sum() / normalizer

    raise ValueError("predictions and targets must share the same shape")


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: Combined,
    dataloader: Iterable[tuple[Tensor, Tensor, Tensor]],
    optimizer: Optimizer,
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    device: torch.device,
    *,
    amp_enabled: bool = False,
    scaler: Optional[GradScaler] = None,
    progress_bar: Optional[tqdm] = None,
    progress_update_by: str = "batches",
) -> float:
    model.train()
    running_loss = 0.0
    batches = 0
    use_amp = amp_enabled and device.type == "cuda"
    scaler = scaler or _create_grad_scaler(use_amp)

    for frames, metadata, targets in dataloader:
        frames = frames.to(device, non_blocking=use_amp)
        metadata = metadata.to(device, non_blocking=use_amp)
        targets = targets.to(device, non_blocking=use_amp)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            predictions, _, visibility = model(frames, metadata)
            loss = loss_fn(predictions, targets, visibility)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        batches += 1

        if progress_bar is not None:
            if progress_update_by == "samples" and hasattr(frames, "shape") and frames.shape:
                increment = int(frames.shape[0])
            else:
                increment = 1
            progress_bar.update(increment)
            avg_loss = running_loss / batches
            progress_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

    average_loss = running_loss / max(batches, 1)
    if progress_bar is not None:
        progress_bar.set_postfix({"avg_loss": f"{average_loss:.4f}"})

    return average_loss


def evaluate(
    model: Combined,
    dataloader: Iterable[tuple[Tensor, Tensor, Tensor]],
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    device: torch.device,
    *,
    amp_enabled: bool = False,
) -> float:
    model.eval()
    running_loss = 0.0
    batches = 0
    use_amp = amp_enabled and device.type == "cuda"

    with torch.no_grad():
        for frames, metadata, targets in dataloader:
            frames = frames.to(device, non_blocking=use_amp)
            metadata = metadata.to(device, non_blocking=use_amp)
            targets = targets.to(device, non_blocking=use_amp)

            with _autocast_context(device, use_amp):
                predictions, _, visibility = model(frames, metadata)
                loss = loss_fn(predictions, targets, visibility)

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
    use_amp: Optional[bool] = None,
    grad_scaler: Optional[GradScaler] = None,
    profile_epoch: bool = False,
) -> Dict[str, list]:
    """Train a combined model using :class:`RandomFaceWindowDataset` samples.

    The dataset itself drives iteration, mirroring the ``RandomFaceWindowDataset``
    interface instead of wrapping it in a :class:`~torch.utils.data.DataLoader`.
    This keeps the dataset's multiprocessing pool and tensor configuration
    intact while still allowing the training loop to treat the object like a
    standard batch iterable.  The ``pin_memory`` argument is retained for API
    compatibility but ignored because no external ``DataLoader`` is built.
    """

    if train_dataset is None:
        train_dataset = RandomFaceWindowDataset(**(train_dataset_kwargs or {}))
    if val_dataset is None and val_dataset_kwargs is not None:
        val_dataset = RandomFaceWindowDataset(**val_dataset_kwargs)

    resolved_device = _resolve_device(device)
    amp_enabled = bool(use_amp) if use_amp is not None else resolved_device.type == "cuda"
    amp_enabled = amp_enabled and resolved_device.type == "cuda"
    if resolved_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        cuda_backends = getattr(torch.backends, "cuda", None)
        if cuda_backends is not None:
            matmul_backend = getattr(cuda_backends, "matmul", None)
            if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
                matmul_backend.allow_tf32 = True  # type: ignore[attr-defined]
        cudnn_backend = getattr(torch.backends, "cudnn", None)
        if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
            cudnn_backend.allow_tf32 = True  # type: ignore[attr-defined]
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("medium")

    model = model or Combined()
    model.to(resolved_device)

    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = loss_fn or masked_mse_loss
    scaler = grad_scaler or _create_grad_scaler(amp_enabled)

    if train_dataset.batch_size != batch_size:
        train_dataset.set_batch_size(batch_size)
    train_dataset.set_num_processes(num_workers)
    train_loader: Iterable[tuple[Tensor, Tensor, Tensor]] = train_dataset

    val_loader: Optional[Iterable[tuple[Tensor, Tensor, Tensor]]] = None
    if val_dataset is not None:
        if val_dataset.batch_size != batch_size:
            val_dataset.set_batch_size(batch_size)
        val_dataset.set_num_processes(num_workers)
        val_loader = val_dataset

    history: Dict[str, list] = {"train_loss": []}
    if val_loader is not None:
        history["val_loss"] = []

    progress_bar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch_idx in progress_bar:
        epoch_bar_kwargs = {
            "desc": f"Epoch {epoch_idx}",
            "unit": "batch",
            "leave": False,
            "position": 1,
        }
        try:
            total_batches = len(train_loader)
        except TypeError:
            total_batches = None
        progress_update_by = "batches"
        dataset_epoch_total = None
        if hasattr(train_loader, "epoch_size"):
            try:
                dataset_epoch_total = int(getattr(train_loader, "epoch_size"))
            except (TypeError, ValueError):
                dataset_epoch_total = None
        elif hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "epoch_size"):
            try:
                dataset_epoch_total = int(getattr(train_loader.dataset, "epoch_size"))
            except (TypeError, ValueError):
                dataset_epoch_total = None
        if dataset_epoch_total is not None and dataset_epoch_total > 0:
            epoch_bar_kwargs["total"] = dataset_epoch_total
            progress_update_by = "samples"
        elif total_batches is not None:
            epoch_bar_kwargs["total"] = total_batches

        with tqdm(**epoch_bar_kwargs) as epoch_progress:
            if profile_epoch and epoch_idx == 1:
                train_loss, profiler_lines, profiler_tables = _profile_train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    loss_fn,
                    resolved_device,
                    amp_enabled=amp_enabled,
                    scaler=scaler,
                    progress_bar=epoch_progress,
                    progress_update_by=progress_update_by,
                )
                for line in profiler_lines:
                    tqdm.write(line)
                for table in profiler_tables:
                    tqdm.write(table)
            else:
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    loss_fn,
                    resolved_device,
                    amp_enabled=amp_enabled,
                    scaler=scaler,
                    progress_bar=epoch_progress,
                    progress_update_by=progress_update_by,
                )
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = evaluate(
                model,
                val_loader,
                loss_fn,
                resolved_device,
                amp_enabled=amp_enabled,
            )
            history["val_loss"].append(val_loss)
            progress_bar.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"})
        else:
            progress_bar.set_postfix({"train_loss": f"{train_loss:.4f}"})

        if scheduler is not None:
            scheduler.step()

    return history


def _create_grad_scaler(enabled: bool) -> GradScaler:
    kwargs = {"enabled": enabled}
    if _AMP_REQUIRES_DEVICE_TYPE:
        kwargs["device_type"] = "cuda"

    try:
        return GradScaler(**kwargs)
    except TypeError as exc:
        if "device_type" not in kwargs:
            raise
        # Older torch.amp versions expose GradScaler but do not accept the
        # device_type argument. Retry without it so training still works.
        kwargs = dict(kwargs)
        kwargs.pop("device_type", None)
        try:
            return GradScaler(**kwargs)
        except TypeError:
            raise exc


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if _AMP_REQUIRES_DEVICE_TYPE:
        return _autocast(device_type=device.type)
    return _autocast()


def _profile_train_epoch(
    model: Combined,
    dataloader: Iterable[tuple[Tensor, Tensor, Tensor]],
    optimizer: Optimizer,
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    device: torch.device,
    *,
    amp_enabled: bool,
    scaler: GradScaler,
    progress_bar: Optional[tqdm] = None,
    progress_update_by: str = "batches",
) -> tuple[float, list[str], List[str]]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    profile_kwargs = dict(activities=activities, record_shapes=True, profile_memory=False)
    profiler_supports_stack = False
    if "with_stack" in inspect.signature(torch_profile).parameters:
        profile_kwargs["with_stack"] = True
        profiler_supports_stack = True

    model.train()
    use_amp = amp_enabled and device.type == "cuda"
    stage_totals = {
        "data_wait": 0.0,
        "h2d_transfer": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
    }

    if isinstance(dataloader, RandomFaceWindowDataset):
        dataset = dataloader
        dataset_instrumented = True
        dataset_worker_warning = False
    else:
        dataset = getattr(dataloader, "dataset", None)
        dataset_instrumented = isinstance(dataset, RandomFaceWindowDataset)
        dataset_worker_warning = dataset_instrumented and getattr(dataloader, "num_workers", 0) > 0
    dataset_timings: Dict[str, Dict[str, float]] = {}
    if dataset_instrumented:
        dataset.enable_timing(reset=True)

    start_time = perf_counter()
    try:
        with torch_profile(**profile_kwargs) as profiler:
            running_loss = 0.0
            batches = 0
            batch_wait_start = perf_counter()

            for frames, metadata, targets in dataloader:
                stage_totals["data_wait"] += perf_counter() - batch_wait_start

                transfer_start = perf_counter()
                frames = frames.to(device, non_blocking=use_amp)
                metadata = metadata.to(device, non_blocking=use_amp)
                targets = targets.to(device, non_blocking=use_amp)
                stage_totals["h2d_transfer"] += perf_counter() - transfer_start

                optimizer.zero_grad(set_to_none=True)

                forward_start = perf_counter()
                with _autocast_context(device, use_amp):
                    predictions, _, visibility = model(frames, metadata)
                    loss = loss_fn(predictions, targets, visibility)
                stage_totals["forward"] += perf_counter() - forward_start

                backward_start = perf_counter()
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    stage_totals["backward"] += perf_counter() - backward_start

                    optimizer_start = perf_counter()
                    scaler.step(optimizer)
                    stage_totals["optimizer"] += perf_counter() - optimizer_start
                    scaler.update()
                else:
                    loss.backward()
                    stage_totals["backward"] += perf_counter() - backward_start

                    optimizer_start = perf_counter()
                    optimizer.step()
                    stage_totals["optimizer"] += perf_counter() - optimizer_start

                running_loss += loss.item()
                batches += 1
                if progress_bar is not None:
                    if progress_update_by == "samples" and hasattr(frames, "shape") and frames.shape:
                        increment = int(frames.shape[0])
                    else:
                        increment = 1
                    progress_bar.update(increment)
                    avg_loss = running_loss / batches
                    progress_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
                batch_wait_start = perf_counter()

            loss = running_loss / max(batches, 1)
    finally:
        if dataset_instrumented:
            dataset_timings = dataset.collect_timings(reset=True)
            dataset.disable_timing()
    duration = perf_counter() - start_time

    if progress_bar is not None:
        progress_bar.set_postfix({"avg_loss": f"{loss:.4f}"})

    summary = profiler.key_averages().total_average()
    cpu_time_ms = getattr(summary, "cpu_time_total", 0.0)
    cuda_time_ms = getattr(summary, "cuda_time_total", 0.0)

    info_lines = [f"Profiled epoch duration: {duration:.3f}s", f"CPU time (profiler): {cpu_time_ms / 1000.0:.3f}s"]
    if cuda_time_ms:
        info_lines.append(f"CUDA time (profiler): {cuda_time_ms / 1000.0:.3f}s")

    num_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    batch_size = getattr(dataloader, "batch_size", None)

    if duration > 0 and num_batches:
        info_lines.append(f"Batches/sec: {num_batches / duration:.2f}")
        if batch_size:
            info_lines.append(f"Approx. samples/sec: {(num_batches * batch_size) / duration:.2f}")

    if duration > 0:
        stage_labels = {
            "data_wait": "Data wait",
            "h2d_transfer": "Host→device transfer",
            "forward": "Forward pass",
            "backward": "Backward pass",
            "optimizer": "Optimizer step",
        }
        info_lines.append("Stage breakdown:")
        for key, label in stage_labels.items():
            total = stage_totals[key]
            percent = (total / duration) * 100 if duration else 0.0
            avg_ms = (total / num_batches) * 1000 if num_batches else 0.0
            info_lines.append(f"  {label}: {total:.3f}s total ({percent:.1f}%), avg {avg_ms:.2f} ms/batch")

    if dataset_timings:
        info_lines.append("RandomFaceWindowDataset timings:")
        for name, stats in sorted(dataset_timings.items(), key=lambda item: item[1]["total"], reverse=True):
            total = stats["total"]
            count = stats["count"]
            avg = stats["avg"]
            info_lines.append(
                f"  {name}: {total:.3f}s total over {count} calls (avg {avg * 1000.0:.2f} ms)"
            )
        if dataset_worker_warning:
            info_lines.append("  Note: worker processes maintain independent timing records.")

    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    tables: List[str] = []
    tables.append("Top operators:\n" + profiler.key_averages().table(sort_by=sort_by, row_limit=10))

    tables.append(
        "Top operators by input shape:\n"
        + profiler.key_averages(group_by_input_shape=True).table(sort_by=sort_by, row_limit=10)
    )

    if profiler_supports_stack:
        tables.append(
            "Hot call stacks:\n"
            + profiler.key_averages(group_by_stack_n=5).table(sort_by=sort_by, row_limit=5)
        )

    return loss, info_lines, tables


__all__ = [
    "train",
    "train_one_epoch",
    "evaluate",
    "masked_mse_loss",
]
