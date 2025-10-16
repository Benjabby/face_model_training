"""Visualization helpers for inspecting :class:`RandomFaceWindowDataset` samples."""

from __future__ import annotations

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


def show(dataset: RandomFaceWindowDataset, batch_number: int = 0, delay: int = 30) -> None:
    """Display a dataset window using the cached face tensors."""

    sample = dataset[batch_number]
    frames: torch.Tensor = sample["frames"]  # type: ignore[assignment]
    metadata: torch.Tensor = sample["face_metadata"]  # type: ignore[assignment]
    dataset_name: str = sample["dataset"]  # type: ignore[assignment]
    video_index: int = sample["video_index"]  # type: ignore[assignment]
    start_frame: int = sample["start_frame"]  # type: ignore[assignment]

    num_frames = int(frames.shape[0])
    for frame_idx in range(num_frames):
        face_tensor = frames[frame_idx]
        if face_tensor.ndim != 3:
            raise ValueError("Frame tensors must be 3-dimensional (C, H, W)")
        if face_tensor.shape[0] < 3:
            raise ValueError("Frame tensors must contain at least three channels")

        face = _tensor_to_bgr(face_tensor[0:3])
        vis = float(metadata[frame_idx, 0].item())
        cx = float(metadata[frame_idx, 1].item())
        cy = float(metadata[frame_idx, 2].item())
        width = float(metadata[frame_idx, 3].item())
        height = float(metadata[frame_idx, 4].item())

        overlay = face.copy()
        h, w = overlay.shape[:2]
        center = (int(round(cx * w)), int(round(cy * h)))
        box_w = max(1, int(round(width * w)))
        box_h = max(1, int(round(height * h)))
        x0 = max(0, center[0] - box_w // 2)
        y0 = max(0, center[1] - box_h // 2)
        x1 = min(w - 1, center[0] + box_w // 2)
        y1 = min(h - 1, center[1] + box_h // 2)

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

        cv2.imshow("RandomFaceWindowDataset - Face", overlay)
        key = cv2.waitKey(delay)
        if key in (27, ord("q")):
            break

    cv2.destroyWindow("RandomFaceWindowDataset - Face")


__all__ = ["show"]
