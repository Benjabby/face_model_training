"""Visualization helpers for inspecting :class:`RandomFaceWindowDataset` samples."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from ..datasets.face_window_dataset import RandomFaceWindowDataset


def _find_video_entry(dataset: RandomFaceWindowDataset, name: str, index: int):
    for entry in dataset._videos:  # type: ignore[attr-defined]
        if entry.dataset_name == name and entry.dataset_index == index:
            return entry
    raise ValueError(f"Unable to locate video entry for {name}[{index}]")


def _tensor_to_bgr(frame_tensor: torch.Tensor) -> np.ndarray:
    frame = frame_tensor.detach().cpu().numpy()
    frame = np.clip(frame, 0.0, 1.0)
    frame = (frame * 255.0).astype(np.uint8)
    frame = np.transpose(frame, (1, 2, 0))
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def show(dataset: RandomFaceWindowDataset, batch_number: int = 0, delay: int = 30) -> None:
    """Display a dataset window using OpenCV windows."""

    sample = dataset[batch_number]
    frames: torch.Tensor = sample["frames"]  # type: ignore[assignment]
    metadata: torch.Tensor = sample["face_metadata"]  # type: ignore[assignment]
    dataset_name: str = sample["dataset"]  # type: ignore[assignment]
    video_index: int = sample["video_index"]  # type: ignore[assignment]
    start_frame: int = sample["start_frame"]  # type: ignore[assignment]

    entry = _find_video_entry(dataset, dataset_name, video_index)
    capture = cv2.VideoCapture(entry.video_path)
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        for frame_idx in range(frames.shape[0]):
            success, full_frame = capture.read()
            if not success:
                break

            visibility = float(metadata[frame_idx, 0].item())
            max_dim = float(max(full_frame.shape[0], full_frame.shape[1]))
            width = float(metadata[frame_idx, 3].item()) * max_dim
            height = float(metadata[frame_idx, 4].item()) * max_dim
            cx = float(metadata[frame_idx, 1].item()) * max_dim
            cy = float(metadata[frame_idx, 2].item()) * max_dim
            x0 = int(round(cx - width / 2.0))
            y0 = int(round(cy - height / 2.0))
            x1 = int(round(cx + width / 2.0))
            y1 = int(round(cy + height / 2.0))

            h, w = full_frame.shape[:2]
            x0_clamped = max(0, min(w - 1, x0))
            y0_clamped = max(0, min(h - 1, y0))
            x1_clamped = max(0, min(w - 1, x1))
            y1_clamped = max(0, min(h - 1, y1))

            if visibility > 0.0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(full_frame, (x0_clamped, y0_clamped), (x1_clamped, y1_clamped), color, 2)

            text = f"vis: {visibility:.2f}"
            text_origin = (x0_clamped, max(20, y0_clamped - 10))
            cv2.putText(
                full_frame,
                text,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            face = _tensor_to_bgr(frames[frame_idx, 0:3])

            cv2.imshow("RandomFaceWindowDataset - Full", full_frame)
            cv2.imshow("RandomFaceWindowDataset - Face", face)
            key = cv2.waitKey(delay)
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyWindow("RandomFaceWindowDataset - Full")
        cv2.destroyWindow("RandomFaceWindowDataset - Face")


__all__ = ["show"]
