"""Utilities for creating random face-window datasets for training models.

This module provides a :class:`RandomFaceWindowDataset` that samples fixed-size
temporal windows of cropped facial regions from multiple physiological video
datasets.  Each sample is composed of ``window_size`` consecutive frames and is
designed to integrate with PyTorch's :class:`~torch.utils.data.DataLoader` in
order to deliver batches shaped ``(B, window_size, 3, image_size, image_size)``.

The dataset randomly chooses a source video from the configured datasets and a
valid starting frame for every request, enabling arbitrarily long epochs without
repeating windows deterministically.  Two optional augmentation stages are
supported: pre-face-detection transforms that act on the full frame, and
post-face-detection transforms that operate on the cropped face prior to
conversion to tensors.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset

from .dataset import CameraData
from .lgi_ppgi import LGI_PPGI
from .pure import PURE
from .ubfc import UBFC1, UBFC2


ArrayLike = Union[np.ndarray, "torch.Tensor", Image.Image]
TransformType = Union[Callable[[ArrayLike], ArrayLike], Sequence[Callable[[ArrayLike], ArrayLike]]]


@dataclass
class _VideoEntry:
    """Container describing a usable video file within a dataset."""

    dataset_name: str
    dataset_index: int
    video_path: str
    num_frames: int


class RandomFaceWindowDataset(TorchDataset):
    """Dataset that samples random facial windows from multiple sources.

    Parameters
    ----------
    lgi_dir, pure_dir, ubfc1_dir, ubfc2_dir:
        Base directories for the supported datasets.  Any ``None`` entry is
        ignored, allowing callers to work with a subset of datasets.
    window_size:
        Number of consecutive frames in every sample window (default: ``360``).
    image_size:
        Spatial size used for the cropped facial region (default: ``112``).
    epoch_size:
        Length reported by :meth:`__len__`, enabling arbitrarily defined epoch
        lengths (default: ``1000``).
    pre_face_transforms, post_face_transforms:
        Optional callables (or sequences of callables) applied before and after
        face detection respectively.  The callables must accept and return
        ``numpy`` arrays, ``torch`` tensors, or ``PIL`` images representing a
        single frame.
    face_detector:
        Optional custom face detector.  When omitted, OpenCV's Haar cascade is
        used.  The detector must expose either a ``detect`` method returning a
        list of bounding boxes or a ``detectMultiScale`` method mirroring the
        OpenCV API.
    seed:
        Optional seed used to build a deterministic per-index random generator.
    """

    def __init__(
        self,
        *,
        lgi_dir: Optional[str] = r"E:\face_datasets\LGI-PPGI",
        pure_dir: Optional[str] = r"E:\face_datasets\PURE",
        ubfc1_dir: Optional[str] = r"E:\face_datasets\UBFC1",
        ubfc2_dir: Optional[str] = r"E:\face_datasets\UBFC2",
        window_size: int = 360,
        image_size: int = 112,
        epoch_size: int = 1000,
        pre_face_transforms: Optional[TransformType] = None,
        post_face_transforms: Optional[TransformType] = None,
        face_detector: Optional[object] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.window_size = window_size
        self.image_size = image_size
        self._epoch_size = epoch_size
        self.pre_face_transforms = pre_face_transforms
        self.post_face_transforms = post_face_transforms
        self.seed = seed

        self.face_detector = face_detector or cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )

        self.datasets: Dict[str, object] = {}
        if lgi_dir:
            self.datasets["LGI_PPGI"] = LGI_PPGI(lgi_dir)
        if pure_dir:
            self.datasets["PURE"] = PURE(pure_dir)
        if ubfc1_dir:
            self.datasets["UBFC1"] = UBFC1(ubfc1_dir)
        if ubfc2_dir:
            self.datasets["UBFC2"] = UBFC2(ubfc2_dir)

        self._videos: List[_VideoEntry] = []
        self._prepare_video_entries()

        if not self._videos:
            raise RuntimeError("No usable videos found for the configured datasets")

    # ------------------------------------------------------------------
    # PyTorch dataset API
    def __len__(self) -> int:
        return self._epoch_size

    def __getitem__(self, index: int) -> Dict[str, Union[str, int, torch.Tensor]]:
        rng = self._get_rng(index)
        entry = rng.choice(self._videos)
        max_start = entry.num_frames - self.window_size
        if max_start <= 0:
            raise RuntimeError(
                f"Video '{entry.video_path}' does not contain enough frames for a window"
            )
        start_frame = rng.randint(0, max_start)
        frames = self._read_face_window(entry.video_path, start_frame)
        return {
            "frames": frames,
            "dataset": entry.dataset_name,
            "video_index": entry.dataset_index,
            "start_frame": start_frame,
        }

    # ------------------------------------------------------------------
    # Public helpers
    @property
    def epoch_size(self) -> int:
        return self._epoch_size

    def set_epoch_size(self, size: int) -> None:
        if size <= 0:
            raise ValueError("Epoch size must be positive")
        self._epoch_size = size

    @property
    def available_datasets(self) -> Sequence[str]:
        return tuple(self.datasets.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    def _get_rng(self, index: int) -> random.Random:
        if self.seed is None:
            return random
        return random.Random(self.seed + index)

    def _prepare_video_entries(self) -> None:
        for name, dataset in self.datasets.items():
            for video_idx in range(len(dataset)):
                video_path = dataset.get_video_path(video_idx)
                entry = self._build_video_entry(name, video_idx, video_path)
                if entry is not None:
                    self._videos.append(entry)

    def _build_video_entry(
        self, dataset_name: str, dataset_index: int, video_path: str
    ) -> Optional[_VideoEntry]:
        camera = CameraData.create(video_path)
        try:
            if camera.nframes < self.window_size:
                return None
            return _VideoEntry(
                dataset_name=dataset_name,
                dataset_index=dataset_index,
                video_path=video_path,
                num_frames=camera.nframes,
            )
        finally:
            camera.close()

    def _read_face_window(self, video_path: str, start_frame: int) -> torch.Tensor:
        capture = cv2.VideoCapture(video_path)
        try:
            if start_frame > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames: List[torch.Tensor] = []
            last_bbox: Optional[Tuple[int, int, int, int]] = None

            for _ in range(self.window_size):
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(
                        f"Failed to read frame {start_frame + len(frames)} from '{video_path}'"
                    )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = self._apply_pre_transforms(frame_rgb)
                bbox = self._detect_face(frame_rgb, last_bbox)
                last_bbox = bbox
                face = self._extract_face(frame_rgb, bbox)
                face = self._apply_post_transforms(face)
                frames.append(face)

            return torch.stack(frames, dim=0)
        finally:
            capture.release()

    def _apply_pre_transforms(self, frame: np.ndarray) -> np.ndarray:
        transformed = self._apply_transforms(self.pre_face_transforms, frame)
        return self._ensure_uint8_np(transformed)

    def _apply_post_transforms(self, face: np.ndarray) -> torch.Tensor:
        transformed = self._apply_transforms(self.post_face_transforms, face)
        if isinstance(transformed, torch.Tensor):
            tensor = transformed
        elif isinstance(transformed, Image.Image):
            tensor = torch.from_numpy(np.array(transformed, dtype=np.uint8))
            tensor = tensor.permute(2, 0, 1)
        else:
            np_face = np.asarray(transformed, dtype=np.uint8)
            if np_face.ndim == 2:
                np_face = np.repeat(np_face[..., None], 3, axis=2)
            tensor = torch.from_numpy(np_face).permute(2, 0, 1)

        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        return tensor

    def _apply_transforms(self, transforms: Optional[TransformType], data: ArrayLike) -> ArrayLike:
        if transforms is None:
            return data
        if isinstance(transforms, Sequence) and not isinstance(transforms, (str, bytes)):
            result: ArrayLike = data
            for transform in transforms:
                result = transform(result)
            return result
        return transforms(data)

    def _detect_face(
        self,
        frame_rgb: np.ndarray,
        previous_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[int, int, int, int]:
        detector = self.face_detector
        bbox: Optional[Tuple[int, int, int, int]] = None

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if hasattr(detector, "detect"):
            detections = detector.detect(frame_rgb)
        else:
            detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if isinstance(detections, np.ndarray):
            detections_list: List[Tuple[int, int, int, int]] = (
                detections.tolist() if detections.size else []
            )
        elif detections is None:
            detections_list = []
        else:
            detections_list = list(detections)

        if detections_list:
            bbox = max(detections_list, key=lambda b: b[2] * b[3])
        elif previous_bbox is not None:
            bbox = previous_bbox

        if bbox is None:
            h, w = frame_rgb.shape[:2]
            side = min(h, w)
            x = (w - side) // 2
            y = (h - side) // 2
            bbox = (x, y, side, side)

        return bbox

    def _extract_face(
        self, frame_rgb: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        x, y, w, h = bbox
        h_img, w_img = frame_rgb.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        face = frame_rgb[y:y2, x:x2]
        if face.size == 0:
            # Fall back to a centered crop if the bounding box is invalid.
            side = min(h_img, w_img)
            cx = w_img // 2
            cy = h_img // 2
            half = side // 2
            face = frame_rgb[cy - half : cy + half, cx - half : cx + half]

        face = cv2.resize(face, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return face

    @staticmethod
    def _ensure_uint8_np(array_like: ArrayLike) -> np.ndarray:
        if isinstance(array_like, torch.Tensor):
            np_array = array_like.detach().cpu().numpy()
        elif isinstance(array_like, Image.Image):
            np_array = np.array(array_like)
        else:
            np_array = np.asarray(array_like)

        if np_array.dtype != np.uint8:
            np_array = np.clip(np_array, 0, 255).astype(np.uint8)
        return np_array


def create_face_window_dataloader(
    batch_size: int,
    *,
    num_workers: int = 0,
    shuffle: bool = False,
    drop_last: bool = True,
    **dataset_kwargs: object,
) -> Tuple[DataLoader, RandomFaceWindowDataset]:
    """Factory helper that builds the dataset and an accompanying dataloader.

    Parameters
    ----------
    batch_size:
        Batch size provided to the :class:`~torch.utils.data.DataLoader`.
    num_workers:
        Number of worker processes for data loading.
    shuffle:
        Whether to enable shuffling at the DataLoader level.  This is disabled by
        default because :class:`RandomFaceWindowDataset` already samples windows
        randomly.
    drop_last:
        Drop the last incomplete batch (defaults to ``True`` for evenly shaped
        training batches).
    dataset_kwargs:
        Additional keyword arguments forwarded to
        :class:`RandomFaceWindowDataset`.

    Returns
    -------
    tuple
        The instantiated dataloader and the underlying dataset instance.
    """

    dataset = RandomFaceWindowDataset(**dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return loader, dataset


__all__ = ["RandomFaceWindowDataset", "create_face_window_dataloader"]

