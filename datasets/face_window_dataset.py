"""Utilities for creating random face-window datasets for training models.

This module provides a :class:`RandomFaceWindowDataset` that samples fixed-size
temporal windows of cropped facial regions from multiple physiological video
datasets.  Each standard sample is composed of ``window_size`` consecutive face
crops and associated metadata, while a dedicated helper can additionally return
scene context suitable for visualization.  The dataset integrates with
PyTorch's :class:`~torch.utils.data.DataLoader` to deliver batches shaped
``(B, window_size, 3, image_size, image_size)`` for the face tensors and
``(B, window_size, 5)`` for the face metadata.

The dataset randomly chooses a source video from the configured datasets and a
valid starting frame for every request, enabling arbitrarily long epochs without
repeating windows deterministically.  Two optional augmentation stages are
supported: pre-face-detection transforms that act on the full frame, and
post-face-detection transforms that operate on the cropped face prior to
conversion to tensors.  Each sample additionally includes a heart-rate sequence
aligned with the returned frame window, producing batches shaped
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
from torch.utils.data import DataLoader, Dataset as TorchDataset, get_worker_info

from .dataset import CameraData
from .lgi_ppgi import LGI_PPGI
from .pure import PURE
from .ubfc import UBFC1, UBFC2
from ..utils.augmentations import (
    default_post_face_transforms,
    default_pre_face_transforms,
)
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
    frame_times: np.ndarray


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
    cache_cameras:
        When ``True`` (default), keep dataset-specific camera handles open per
        worker so repeated samples avoid reopening video streams.  Disable to
        match the previous one-shot behavior.
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
        cache_cameras: bool = True,
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
        if pre_face_transforms is None:
            pre_face_transforms = default_pre_face_transforms()
        if post_face_transforms is None:
            post_face_transforms = default_post_face_transforms(
                window_size=self.window_size
            )
        self.pre_face_transforms = pre_face_transforms
        self.post_face_transforms = post_face_transforms
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._cache_cameras = cache_cameras
        self._camera_cache: Dict[Optional[int], Dict[str, CameraData]] = {}

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

    def __del__(self) -> None:
        try:
            self._close_camera_cache()
        except Exception:
            pass

    def __getstate__(self) -> Dict[str, object]:
        self._close_camera_cache()
        state = self.__dict__.copy()
        state["_camera_cache"] = {}
        return state

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__dict__.update(state)
        if "_camera_cache" not in self.__dict__:
            self._camera_cache = {}

    # ------------------------------------------------------------------
    # PyTorch dataset API
    def __len__(self) -> int:
        return self._epoch_size

    def __getitem__(self, index: int) -> Dict[str, Union[str, int, torch.Tensor]]:
        return self._sample_window(include_context=False)

    def get_window_with_context(
        self, index: int = 0
    ) -> Dict[str, Union[str, int, torch.Tensor]]:
        """Return a sampled window that preserves context frames.

        Parameters
        ----------
        index:
            Optional deterministic index used to seed a temporary random
            generator.  Reusing the same index will yield the same sampled
            window when the dataset was initialized with a fixed seed.  The
            dataset's main random generator remains unaffected.
        """

        rng = self._spawn_index_rng(index)
        return self._sample_window(include_context=True, rng=rng)

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

    def _spawn_index_rng(self, index: int) -> np.random.Generator:
        if index < 0:
            raise ValueError("index must be non-negative")

        if self.seed is not None:
            seed_seq = np.random.SeedSequence(self.seed, spawn_key=(int(index),))
        else:
            seed_seq = np.random.SeedSequence(int(index))

        return np.random.default_rng(seed_seq)

    def _sample_window(
        self,
        *,
        include_context: bool,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Union[str, int, torch.Tensor]]:
        active_rng = self._get_rng() if rng is None else rng
        entry, start_frame = self._select_entry_and_start(active_rng)
        self._reset_transform_state(active_rng)
        (
            frames,
            face_metadata,
            context_frames,
            context_transforms,
        ) = self._read_face_window(
            entry,
            start_frame,
            active_rng,
            include_context=include_context,
        )
        heart_rates = self._slice_heart_rates(entry, start_frame)
        sample: Dict[str, Union[str, int, torch.Tensor]] = {
            "frames": frames,
            "face_metadata": face_metadata,
            "dataset": entry.dataset_name,
            "video_index": entry.dataset_index,
            "start_frame": start_frame,
            "heart_rates": heart_rates,
        }
        if include_context:
            assert context_frames is not None and context_transforms is not None
            sample["context_frames"] = context_frames
            sample["context_transforms"] = context_transforms
        return sample

    def _select_entry_and_start(
        self, rng: np.random.Generator
    ) -> Tuple[_VideoEntry, int]:
        entry_idx = int(rng.integers(0, len(self._videos)))
        entry = self._videos[entry_idx]
        max_start = entry.num_frames - self.window_size
        if max_start <= 0:
            raise RuntimeError(
                f"Video '{entry.video_path}' does not contain enough frames for a window"
            )
        start_frame = int(rng.integers(0, max_start + 1))
        return entry, start_frame

    def _reset_transform_state(self, rng: np.random.Generator) -> None:
        for transform in self._pre_face_transforms + self._post_face_transforms:
            self._invoke_transform_reset(transform, rng)

    def _invoke_transform_reset(
        self, transform: Callable[[ArrayLike], ArrayLike], rng: np.random.Generator
    ) -> None:
        reset = getattr(transform, "reset", None)
        if not callable(reset):
            return
        try:
            signature = inspect.signature(reset)
        except (TypeError, ValueError):
            signature = None

        if signature is None:
            try:
                reset(self.window_size, rng)
                return
            except TypeError:
                try:
                    reset(self.window_size)
                    return
                except TypeError:
                    try:
                        reset(rng)
                        return
                    except TypeError:
                        reset()
                        return

        kwargs = {}
        params = signature.parameters
        if "window_size" in params:
            kwargs["window_size"] = self.window_size
        if "rng" in params:
            kwargs["rng"] = rng
        try:
            reset(**kwargs)
        except TypeError:
            try:
                reset(self.window_size, rng)
            except TypeError:
                try:
                    reset(self.window_size)
                except TypeError:
                    try:
                        reset(rng)
                    except TypeError:
                        reset()

    def _close_camera_cache(self) -> None:
        cache_dict = getattr(self, "_camera_cache", None)
        if not cache_dict:
            return
        for cache in cache_dict.values():
            for camera in cache.values():
                try:
                    camera.close()
                except Exception:
                    continue
        cache_dict.clear()

    def _worker_cache_key(self) -> Optional[int]:
        worker_info = get_worker_info()
        return None if worker_info is None else worker_info.id

    def _get_cached_camera(self, entry: _VideoEntry) -> CameraData:
        cache_key = self._worker_cache_key()
        cache = self._camera_cache.setdefault(cache_key, {})
        camera = cache.get(entry.video_path)
        if camera is None:
            camera = CameraData.create(entry.video_path, timestamps=entry.frame_times)
            cache[entry.video_path] = camera
        camera.reset()
        return camera

    def _invalidate_cached_camera(
        self, entry: _VideoEntry, camera: Optional[CameraData]
    ) -> None:
        if camera is None or not self._cache_cameras:
            return
        cache_key = self._worker_cache_key()
        cache = self._camera_cache.get(cache_key)
        if not cache:
            return
        cached = cache.get(entry.video_path)
        if cached is camera:
            try:
                camera.close()
            finally:
                cache.pop(entry.video_path, None)

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
        try:
            ground_truth, video_times = dataset.load_instance(
                dataset_index, include_video=False
            )
        except Exception:
            return None

        timestamps = None
        if video_times is not None:
            timestamps = np.asarray(video_times, dtype=np.float64)

        camera: Optional[CameraData] = None
        try:
            camera = CameraData.create(video_path, timestamps=timestamps)
        except Exception:
            return None

        try:
            if camera.nframes < self.window_size:
                return None

            frame_times = np.asarray(camera.times)

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
                frame_times=frame_times,
            )
        finally:
            if camera is not None:
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
        self,
        entry: _VideoEntry,
        start_frame: int,
        rng: np.random.Generator,
        *,
        include_context: bool,
        camera: Optional[CameraData] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        owns_camera = False
        cached_camera = False
        if camera is None:
            if self._cache_cameras:
                camera = self._get_cached_camera(entry)
                cached_camera = True
            else:
                camera = CameraData.create(entry.video_path, timestamps=entry.frame_times)
                owns_camera = True
        assert camera is not None  # For type checkers

        try:
            try:
                self._fast_forward_camera(camera, start_frame)
            except Exception:
                if cached_camera:
                    self._invalidate_cached_camera(entry, camera)
                raise

            face_frames = torch.empty(
                (self.window_size, 3, self.image_size, self.image_size),
                dtype=torch.float32,
            )
            metadata = torch.zeros((self.window_size, 5), dtype=torch.float32)
            if include_context:
                context_frames = torch.empty(
                    (self.window_size, 3, self.image_size, self.image_size),
                    dtype=torch.float32,
                )
                context_transforms = torch.zeros((self.window_size, 5), dtype=torch.float32)
            else:
                context_frames = None
                context_transforms = None
            last_bbox: Optional[Tuple[int, int, int, int]] = None

            for frame_idx in range(self.window_size):
                try:
                    frame_bgr, _ = next(camera)
                except StopIteration as exc:
                    if cached_camera:
                        self._invalidate_cached_camera(entry, camera)
                    raise RuntimeError(
                        f"Failed to read frame {start_frame + frame_idx} from '{entry.video_path}'"
                    ) from exc

                frame_bgr = frame_bgr.copy()
                frame_bgr = self._apply_pre_transforms(frame_bgr, rng)
                bbox, visibility = self._detect_face(frame_bgr, last_bbox)
                face_rgb, adjusted_bbox = self._extract_face(frame_bgr, bbox)
                last_bbox = adjusted_bbox
                face_tensor, post_visibility = self._apply_post_transforms(
                    face_rgb, rng
                )
                face_frames[frame_idx].copy_(face_tensor)
                if include_context:
                    (
                        context_tensor,
                        transform_info,
                    ) = self._prepare_context_frame(frame_bgr)
                    assert context_frames is not None and context_transforms is not None
                    context_frames[frame_idx].copy_(context_tensor)
                    context_transforms[frame_idx].copy_(
                        torch.as_tensor(transform_info, dtype=torch.float32)
                    )
                combined_visibility = visibility * post_visibility
                metadata[frame_idx].copy_(
                    self._build_face_metadata(
                        adjusted_bbox,
                        combined_visibility,
                        frame_bgr.shape,
                    )
                )

            return face_frames, metadata, context_frames, context_transforms
        finally:
            if owns_camera:
                camera.close()

    def _fast_forward_camera(self, camera: CameraData, frames_to_skip: int) -> None:
        if frames_to_skip <= 0:
            return
        try:
            camera.skip(frames_to_skip)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to skip {frames_to_skip} frames within camera stream"
            ) from exc

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
    ) -> Tuple[torch.Tensor, float]:
        data: ArrayLike = face_rgb
        blackout = False

        for transform in self._post_face_transform_funcs:
            result = transform(data, rng)
            if isinstance(result, tuple):
                if len(result) != 2:
                    raise ValueError(
                        "Post-face transforms must return (data, visibility) tuples"
                    )
                data, visibility = result
                if self._coerce_visibility_factor(visibility) <= 0.0:
                    blackout = True
            else:
                data = result

        tensor = self._to_normalized_tensor(data)
        visibility_scale = 0.0 if blackout else 1.0
        return tensor, visibility_scale

    def _prepare_context_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[torch.Tensor, Tuple[float, float, float, float, float]]:
        target_size = self.image_size
        frame_h, frame_w = frame_bgr.shape[:2]
        max_dim = max(frame_h, frame_w, 1)
        scale = float(target_size) / float(max_dim)

        new_w = max(1, int(round(frame_w * scale)))
        new_h = max(1, int(round(frame_h * scale)))

        interpolation = (
            cv2.INTER_AREA
            if frame_h > target_size or frame_w > target_size
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=interpolation)

        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        pad_x = max(0, (target_size - new_w) // 2)
        pad_y = max(0, (target_size - new_h) // 2)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        context_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        context_tensor = torch.as_tensor(context_rgb, dtype=torch.float32).permute(2, 0, 1)
        if context_tensor.max() > 1.0:
            context_tensor.mul_(1.0 / 255.0)

        transform_info = (
            scale,
            float(pad_x),
            float(pad_y),
            float(frame_h),
            float(frame_w),
        )
        return context_tensor, transform_info

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
    ) -> Tuple[Tuple[int, int, int, int], float]:
        detector = self.face_detector
        bbox: Optional[Tuple[int, int, int, int]] = None
        visibility = 0.0

        boxes, scores = detector.process(frame_bgr)
        boxes = np.asarray(boxes)
        scores = np.asarray(scores)

        if boxes.size:
            if scores.size:
                best_idx = int(np.argmax(scores))
                visibility = float(scores[best_idx])
            else:
                best_idx = 0
                visibility = 0.0
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

        return bbox, visibility

    def _extract_face(
        self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        x, y, w, h = bbox
        h_img, w_img = frame_bgr.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        w = max(1, x2 - x)
        h = max(1, y2 - y)

        face = frame_bgr[y:y2, x:x2]
        if face.size == 0:
            # Fall back to a centered crop if the bounding box is invalid.
            side = min(h_img, w_img)
            cx = w_img // 2
            cy = h_img // 2
            half = side // 2
            face = frame_bgr[cy - half : cy + half, cx - half : cx + half]
            x = cx - half
            y = cy - half
            w = face.shape[1]
            h = face.shape[0]

        interpolation = (
            cv2.INTER_AREA
            if face.shape[0] > self.image_size or face.shape[1] > self.image_size
            else cv2.INTER_LINEAR
        )
        face = cv2.resize(face, (self.image_size, self.image_size), interpolation=interpolation)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face, (x, y, w, h)

    def _build_face_metadata(
        self,
        bbox: Tuple[int, int, int, int],
        visibility: float,
        frame_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        x, y, w, h = bbox
        frame_h = float(frame_shape[0])
        frame_w = float(frame_shape[1])
        max_dim = max(frame_h, frame_w)
        if max_dim <= 0:
            max_dim = 1.0

        cx = float(x) + float(w) / 2.0
        cy = float(y) + float(h) / 2.0

        metadata = torch.tensor(
            [
                float(visibility),
                cx / max_dim,
                cy / max_dim,
                float(w) / max_dim,
                float(h) / max_dim,
            ],
            dtype=torch.float32,
        )
        metadata[1:].clamp_(0.0, 1.0)
        return metadata

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
    def _coerce_visibility_factor(value: object) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Visibility factors must be scalar tensors")
            return float(value.item())

        if np.isscalar(value):
            return float(value)  # type: ignore[arg-type]

        array = np.asarray(value)
        if array.size != 1:
            raise ValueError("Visibility factors must be scalar values")
        return float(array.item())

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
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        **({"prefetch_factor": 4} if num_workers > 0 else {}),
    )
    return loader, dataset


__all__ = ["RandomFaceWindowDataset", "create_face_window_dataloader"]

