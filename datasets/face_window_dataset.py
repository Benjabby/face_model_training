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
conversion to tensors.  Each sample additionally includes a heart-rate sequence
aligned with the returned frame window, producing batches shaped
``(B, window_size, 3, image_size, image_size)`` for faces and
``(B, window_size)`` for heart rates.
"""

from __future__ import annotations

import inspect
import os
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
from ..utils.face_utils import BaseDetector, select_face_detector


ArrayLike = Union[np.ndarray, "torch.Tensor", Image.Image]
TransformType = Union[Callable[[ArrayLike], ArrayLike], Sequence[Callable[[ArrayLike], ArrayLike]]]


@dataclass
class _VideoEntry:
    """Container describing a usable video file within a dataset."""

    dataset_name: str
    dataset_index: int
    video_path: str
    num_frames: int
    heart_rates: np.ndarray


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
        face detection respectively.  Pre-face transforms receive RGB frames,
        while post-face transforms operate on RGB face crops.  The callables
        must accept and return ``numpy`` arrays, ``torch`` tensors, or ``PIL``
        images representing a single frame.
    face_detector:
        Name of the face detector to use from :mod:`utils.face_utils`
        (default: ``"yunet"``).  Additional keyword arguments can be
        supplied via ``face_detector_kwargs``.
    face_detector_kwargs:
        Optional dictionary of keyword arguments forwarded to the detector
        factory.
    seed:
        Optional seed forwarded to :func:`numpy.random.default_rng` to
        initialize the dataset's random generator.
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
        face_detector: str = "yunet",
        face_detector_kwargs: Optional[Dict[str, object]] = None,
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
        self._pre_face_transforms: Tuple[Callable[[ArrayLike], ArrayLike], ...] = ()
        self._post_face_transforms: Tuple[Callable[[ArrayLike], ArrayLike], ...] = ()
        self._pre_face_transform_funcs: Tuple[
            Callable[[ArrayLike, np.random.Generator], ArrayLike], ...
        ] = ()
        self._post_face_transform_funcs: Tuple[
            Callable[[ArrayLike, np.random.Generator], ArrayLike], ...
        ] = ()
        self.pre_face_transforms = pre_face_transforms
        self.post_face_transforms = post_face_transforms
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        detector_kwargs = dict(face_detector_kwargs or {})
        self.face_detector: BaseDetector = select_face_detector(
            face_detector,
            res_dir=os.path.join(project_root, "utils", "res"),
            **detector_kwargs,
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
        rng = self._get_rng()
        entry_idx = int(rng.integers(0, len(self._videos)))
        entry = self._videos[entry_idx]
        max_start = entry.num_frames - self.window_size
        if max_start <= 0:
            raise RuntimeError(
                f"Video '{entry.video_path}' does not contain enough frames for a window"
            )
        start_frame = int(rng.integers(0, max_start + 1))
        frames = self._read_face_window(entry, start_frame, rng)
        heart_rates = self._slice_heart_rates(entry, start_frame)
        return {
            "frames": frames,
            "dataset": entry.dataset_name,
            "video_index": entry.dataset_index,
            "start_frame": start_frame,
            "heart_rates": heart_rates,
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
    def _get_rng(self) -> np.random.Generator:
        return self.rng

    def _prepare_video_entries(self) -> None:
        for name, dataset in self.datasets.items():
            for video_idx in range(len(dataset)):
                video_path = dataset.get_video_path(video_idx)
                entry = self._build_video_entry(name, dataset, video_idx, video_path)
                if entry is not None:
                    self._videos.append(entry)

    def _build_video_entry(
        self,
        dataset_name: str,
        dataset: object,
        dataset_index: int,
        video_path: str,
    ) -> Optional[_VideoEntry]:
        camera = CameraData.create(video_path)
        try:
            if camera.nframes < self.window_size:
                return None

            frame_times = np.asarray(camera.times)

            try:
                ground_truth, _ = dataset.load_instance(
                    dataset_index, include_video=False
                )
            except Exception:
                return None

            hr_signal = getattr(ground_truth, "HR", None)
            if hr_signal is None:
                return None

            heart_rates = hr_signal.resample(frame_times, method="nearest")
            heart_rates = np.asarray(heart_rates, dtype=np.float32)
            if heart_rates.shape[0] != camera.nframes:
                return None

            return _VideoEntry(
                dataset_name=dataset_name,
                dataset_index=dataset_index,
                video_path=video_path,
                num_frames=camera.nframes,
                heart_rates=heart_rates,
            )
        finally:
            camera.close()

    def _slice_heart_rates(self, entry: _VideoEntry, start_frame: int) -> torch.Tensor:
        end_frame = start_frame + self.window_size
        heart_rates = entry.heart_rates[start_frame:end_frame]
        if heart_rates.shape[0] != self.window_size:
            raise RuntimeError(
                "Heart rate annotations do not cover the requested window"
            )
        return torch.as_tensor(heart_rates, dtype=torch.float32)

    def _read_face_window(
        self, entry: _VideoEntry, start_frame: int, rng: np.random.Generator
    ) -> torch.Tensor:
        capture = cv2.VideoCapture(entry.video_path)
        try:
            if start_frame > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames = torch.empty(
                (self.window_size, 3, self.image_size, self.image_size),
                dtype=torch.float32,
            )
            last_bbox: Optional[Tuple[int, int, int, int]] = None

            for frame_idx in range(self.window_size):
                success, frame_bgr = capture.read()
                if not success:
                    raise RuntimeError(
                        f"Failed to read frame {start_frame + frame_idx} from '{entry.video_path}'"
                    )

                frame_bgr = self._apply_pre_transforms(frame_bgr, rng)
                bbox = self._detect_face(frame_bgr, last_bbox)
                last_bbox = bbox
                face_rgb = self._extract_face(frame_bgr, bbox)
                face_tensor = self._apply_post_transforms(face_rgb, rng)
                frames[frame_idx].copy_(face_tensor)

            return frames
        finally:
            capture.release()

    def _apply_pre_transforms(
        self, frame_bgr: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if not self._pre_face_transform_funcs:
            return self._ensure_uint8_np(frame_bgr)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        transformed = self._apply_transforms(
            self._pre_face_transform_funcs, frame_rgb, rng
        )
        frame_rgb = self._ensure_uint8_np(transformed)
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def _apply_post_transforms(
        self, face_rgb: np.ndarray, rng: np.random.Generator
    ) -> torch.Tensor:
        transformed = self._apply_transforms(
            self._post_face_transform_funcs, face_rgb, rng
        )
        return self._to_normalized_tensor(transformed)

    def _apply_transforms(
        self,
        transforms: Sequence[Callable[[ArrayLike, np.random.Generator], ArrayLike]],
        data: ArrayLike,
        rng: np.random.Generator,
    ) -> ArrayLike:
        result: ArrayLike = data
        for transform in transforms:
            result = transform(result, rng)
        return result

    def _detect_face(
        self,
        frame_bgr: np.ndarray,
        previous_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[int, int, int, int]:
        detector = self.face_detector
        bbox: Optional[Tuple[int, int, int, int]] = None

        boxes, scores = detector.process(frame_bgr)
        boxes = np.asarray(boxes)
        scores = np.asarray(scores)

        if boxes.size:
            if scores.size:
                best_idx = int(np.argmax(scores))
            else:
                best_idx = 0
            x0, y0, x1, y1 = boxes[best_idx]
            w = max(1, int(round(x1 - x0)))
            h = max(1, int(round(y1 - y0)))
            x = int(round(x0))
            y = int(round(y0))
            bbox = (x, y, w, h)
        elif previous_bbox is not None:
            bbox = previous_bbox

        if bbox is None:
            h, w = frame_bgr.shape[:2]
            side = min(h, w)
            x = (w - side) // 2
            y = (h - side) // 2
            bbox = (x, y, side, side)

        return bbox

    def _extract_face(
        self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        x, y, w, h = bbox
        h_img, w_img = frame_bgr.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        face = frame_bgr[y:y2, x:x2]
        if face.size == 0:
            # Fall back to a centered crop if the bounding box is invalid.
            side = min(h_img, w_img)
            cx = w_img // 2
            cy = h_img // 2
            half = side // 2
            face = frame_bgr[cy - half : cy + half, cx - half : cx + half]

        interpolation = (
            cv2.INTER_AREA
            if face.shape[0] > self.image_size or face.shape[1] > self.image_size
            else cv2.INTER_LINEAR
        )
        face = cv2.resize(face, (self.image_size, self.image_size), interpolation=interpolation)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face

    def _normalize_transforms(
        self, transforms: Optional[TransformType]
    ) -> Tuple[Callable[[ArrayLike], ArrayLike], ...]:
        if transforms is None:
            return ()
        if isinstance(transforms, Sequence) and not isinstance(transforms, (str, bytes)):
            return tuple(transforms)
        return (transforms,)

    def _wrap_transforms(
        self, transforms: Sequence[Callable[[ArrayLike], ArrayLike]]
    ) -> Tuple[Callable[[ArrayLike, np.random.Generator], ArrayLike], ...]:
        wrapped: List[Callable[[ArrayLike, np.random.Generator], ArrayLike]] = []
        for transform in transforms:
            wrapped.append(self._wrap_single_transform(transform))
        return tuple(wrapped)

    def _wrap_single_transform(
        self, transform: Callable[[ArrayLike], ArrayLike]
    ) -> Callable[[ArrayLike, np.random.Generator], ArrayLike]:
        try:
            signature = inspect.signature(transform)
        except (TypeError, ValueError):
            signature = None

        def caller(data: ArrayLike, rng: np.random.Generator) -> ArrayLike:
            if signature is None:
                try:
                    return transform(data, rng=rng)  # type: ignore[call-arg]
                except TypeError as exc:
                    message = str(exc)
                    if "unexpected keyword argument" in message and "'rng'" in message:
                        return transform(data)  # type: ignore[misc]
                    raise

            params = list(signature.parameters.values())
            has_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in params
            )
            accepts_rng = "rng" in signature.parameters or has_kwargs
            accepts_data = any(
                param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for param in params
            )

            if accepts_rng:
                if accepts_data:
                    return transform(data, rng=rng)  # type: ignore[call-arg]
                return transform(rng=rng)  # type: ignore[call-arg]

            if accepts_data:
                return transform(data)  # type: ignore[misc]

            return transform()  # type: ignore[misc]

        return caller

    @property
    def pre_face_transforms(self) -> Optional[TransformType]:
        return getattr(self, "_pre_face_transforms_config", None)

    @pre_face_transforms.setter
    def pre_face_transforms(self, transforms: Optional[TransformType]) -> None:
        self._pre_face_transforms_config = transforms
        normalized = self._normalize_transforms(transforms)
        self._pre_face_transforms = normalized
        self._pre_face_transform_funcs = self._wrap_transforms(normalized)

    @property
    def post_face_transforms(self) -> Optional[TransformType]:
        return getattr(self, "_post_face_transforms_config", None)

    @post_face_transforms.setter
    def post_face_transforms(self, transforms: Optional[TransformType]) -> None:
        self._post_face_transforms_config = transforms
        normalized = self._normalize_transforms(transforms)
        self._post_face_transforms = normalized
        self._post_face_transform_funcs = self._wrap_transforms(normalized)

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
        elif not np_array.flags["C_CONTIGUOUS"]:
            np_array = np.ascontiguousarray(np_array)
        return np_array

    @staticmethod
    def _to_normalized_tensor(face_like: ArrayLike) -> torch.Tensor:
        if isinstance(face_like, torch.Tensor):
            tensor = face_like.detach()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            if tensor.ndim != 3:
                raise ValueError("Face tensors must be 3-dimensional")
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            if tensor.dtype != torch.float32:
                tensor = tensor.to(dtype=torch.float32)
            max_val = tensor.max()
            if max_val > 1.0:
                tensor = tensor / 255.0
            return tensor

        if isinstance(face_like, Image.Image):
            np_face = np.array(face_like)
        else:
            np_face = np.asarray(face_like)

        if np_face.ndim == 2:
            np_face = np_face[..., None]
        if np_face.shape[-1] == 1:
            np_face = np.repeat(np_face, 3, axis=2)
        elif np_face.shape[-1] != 3:
            raise ValueError("Face arrays must have 1 or 3 channels")

        np_face = np.ascontiguousarray(np_face)
        tensor = torch.as_tensor(np_face, dtype=torch.float32).permute(2, 0, 1)
        if np_face.dtype == np.uint8:
            tensor.mul_(1.0 / 255.0)
        elif tensor.max() > 1.0:
            tensor.mul_(1.0 / 255.0)
        return tensor


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

