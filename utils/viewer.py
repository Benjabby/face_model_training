"""Visualization helpers for inspecting :class:`RandomFaceWindowDataset` samples."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from ..datasets.face_window_dataset import RandomFaceWindowDataset


def _tensor_to_bgr(frame_tensor: torch.Tensor) -> np.ndarray:
    frame = frame_tensor.detach().cpu().numpy()
    frame = np.clip(frame, 0.0, 1.0)
    frame = (frame * 255.0).astype(np.uint8)
    frame = np.transpose(frame, (1, 2, 0))
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def show(
    dataset: RandomFaceWindowDataset,
    batch_number: int = 0,
    batch_index: Optional[int] = None,
    dataset_name: Optional[str] = None,
    dataset_index: Optional[int] = None,
    video_index: Optional[int] = None,
    delay: int = 30,
) -> None:
    """Display a dataset window using the cached face tensors.

    Parameters
    ----------
    dataset:
        Dataset instance to sample windows from.
    batch_number:
        Legacy positional argument preserved for backwards compatibility.  When
        ``batch_index`` is not provided, this value selects the sampled window.
    batch_index:
        Explicit index identifying the deterministic window to visualize.  When
        provided it overrides ``batch_number`` and is forwarded to
        :meth:`RandomFaceWindowDataset.get_window_with_context`.
    dataset_name:
        Optional dataset identifier restricting the sampled window to the
        specified source dataset.  Mutually exclusive with ``dataset_index``.
    dataset_index:
        Optional index selecting a dataset from the dataset's
        ``available_datasets`` attribute.  When provided it overrides
        ``dataset_name``.
    video_index:
        Optional index identifying a specific video within the chosen dataset.
    delay:
        Milliseconds to wait between frame updates when displaying the window.
    """

    effective_index = batch_number if batch_index is None else batch_index
    if dataset_index is not None:
        available = dataset.available_datasets
        if not available:
            raise ValueError("Dataset does not expose any available dataset names")
        if dataset_index < 0 or dataset_index >= len(available):
            raise IndexError(
                f"dataset_index {dataset_index} out of range for {len(available)} datasets"
            )
        resolved_name = available[dataset_index]
        if dataset_name is not None and dataset_name != resolved_name:
            raise ValueError(
                "dataset_name does not match the dataset resolved from dataset_index"
            )
        dataset_name = resolved_name
    elif dataset_name is not None:
        available = dataset.available_datasets
        if dataset_name not in available:
            raise ValueError(
                f"dataset_name '{dataset_name}' not found in available datasets"
            )

    sample = dataset.get_window_with_context(
        effective_index,
        dataset_name=dataset_name,
        video_index=video_index,
    )
    frames: torch.Tensor = sample["frames"]  # type: ignore[assignment]
    metadata: torch.Tensor = sample["face_metadata"]  # type: ignore[assignment]
    context_frames: torch.Tensor = sample.get(  # type: ignore[assignment]
        "context_frames"
    )
    dataset_name: str = sample["dataset"]  # type: ignore[assignment]
    video_index: int = sample["video_index"]  # type: ignore[assignment]
    start_frame: int = sample["start_frame"]  # type: ignore[assignment]

    if context_frames is None:
        raise KeyError(
            "Sample does not include context frames required for visualization"
        )

    num_frames = int(frames.shape[0])
    for frame_idx in range(num_frames):
        face_tensor = frames[frame_idx]
        context_tensor = context_frames[frame_idx]
        if face_tensor.ndim != 3:
            raise ValueError("Frame tensors must be 3-dimensional (C, H, W)")
        if face_tensor.shape[0] < 3:
            raise ValueError("Frame tensors must contain at least three channels")

        if context_tensor.ndim != 3:
            raise ValueError("Context frames must be 3-dimensional (C, H, W)")
        if context_tensor.shape[0] < 3:
            raise ValueError("Context frames must contain at least three channels")

        face = _tensor_to_bgr(face_tensor[0:3])
        context = _tensor_to_bgr(context_tensor[0:3])
        vis = float(metadata[frame_idx, 0].item())
        cx = float(metadata[frame_idx, 1].item())
        cy = float(metadata[frame_idx, 2].item())
        width = float(metadata[frame_idx, 3].item())
        height = float(metadata[frame_idx, 4].item())

        frame_h = context_tensor.shape[1]
        frame_w = context_tensor.shape[2]
        max_dim = max(float(frame_h), float(frame_w), 1.0)

        cx_px = cx * max_dim
        cy_px = cy * max_dim
        box_w = width * max_dim
        box_h = height * max_dim
        x0 = int(round(cx_px - box_w / 2.0))
        y0 = int(round(cy_px - box_h / 2.0))
        x1 = int(round(cx_px + box_w / 2.0))
        y1 = int(round(cy_px + box_h / 2.0))

        overlay = context.copy()
        h, w = overlay.shape[:2]
        x0 = int(np.clip(x0, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))

        color = (0, 255, 0) if vis > 0.0 else (0, 0, 255)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 1)

        info_lines = [
            f"dataset: {dataset_name}",
            f"video: {video_index}",
            f"frame: {start_frame + frame_idx}",
            f"visibility: {vis:.2f}",
            f"center: ({cx:.3f}, {cy:.3f})",
            f"size: ({width:.3f}, {height:.3f})",
        ]

        y_cursor = 18
        for line in info_lines:
            cv2.putText(
                overlay,
                line,
                (8, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            y_cursor += 18

        cv2.imshow("RandomFaceWindowDataset - Original", overlay)

        face_display = face.copy()
        cv2.putText(
            face_display,
            f"visibility: {vis:.2f}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("RandomFaceWindowDataset - Face", face_display)
        key = cv2.waitKey(delay)
        if key in (27, ord("q")):
            break

    cv2.destroyWindow("RandomFaceWindowDataset - Original")
    cv2.destroyWindow("RandomFaceWindowDataset - Face")


__all__ = ["show"]
