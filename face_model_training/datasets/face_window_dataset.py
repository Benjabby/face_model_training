"""Utilities for creating random face-window datasets for training models.

This module provides a :class:`RandomFaceWindowDataset` that samples fixed-size
temporal windows of cropped facial regions from multiple physiological video
datasets.  Each standard sample is composed of ``window_size`` consecutive face
crops and associated metadata, while a dedicated helper can additionally return
scene context suitable for visualization.  The dataset integrates with
PyTorch's :class:`~torch.utils.data.DataLoader` to deliver batches shaped
``(B, window_size, 6, image_size, image_size)`` for the face tensors and
``(B, window_size, 5)`` for the face metadata.

The face tensor's channel dimension is organized such that the first three
channels contain the normalized RGB data while channels 3-5 (zero-indexed)
store the per-frame differences (current RGB frame minus the previous frame,
with zeros for the initial timestep).

The dataset randomly chooses a source video from the configured datasets and a
valid starting frame for every request, enabling arbitrarily long epochs without
repeating windows deterministically.  Two optional augmentation stages are
supported: pre-face-detection transforms that act on the full frame, and
post-face-detection transforms that operate on the cropped face prior to
conversion to tensors.  Each sample additionally includes a heart-rate target
aligned with the returned frame window, delivered either as the full
``(B, window_size)`` sequence or as a scalar ``(B,)`` mean depending on the
configured output mode.
"""

from __future__ import annotations

import copy
import inspect
import multiprocessing as mp
import os
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset, get_worker_info
from torch.utils.data._utils.collate import default_collate

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
    heart_rate_as_scalar:
        When ``True`` (default), return the mean heart rate for each sampled
        window as a scalar tensor instead of the full per-frame signal.
    batch_size:
        Default number of windows returned by :meth:`get_batch`.
    num_processes:
        Optional number of worker processes used by :meth:`get_batch`.  When
        ``None`` (default) the worker count is derived from
        :func:`os.cpu_count` and the configured batch size.
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
        lgi_dir: Optional[str] = r"C:\Users\Ben\Desktop\face_datasets\LGI-PPGI",
        pure_dir: Optional[str] = r"C:\Users\Ben\Desktop\face_datasets\PURE",
        ubfc1_dir: Optional[str] = r"C:\Users\Ben\Desktop\face_datasets\UBFC1",
        ubfc2_dir: Optional[str] = r"C:\Users\Ben\Desktop\face_datasets\UBFC2",
        window_size: int = 360,
        image_size: int = 112,
        epoch_size: int = 1000,
        pre_face_transforms: Optional[TransformType] = None,
        post_face_transforms: Optional[TransformType] = None,
        face_detector: str = "yunet",
        face_detector_kwargs: Optional[Dict[str, object]] = None,
        heart_rate_as_scalar: bool = True,
        batch_size: int = 1,
        num_processes: Optional[int] = None,
        seed: Optional[int] = None,
        cache_cameras: bool = True,
        augment_scale = 0.5
    ) -> None:
        super().__init__()

        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_processes is not None and int(num_processes) < 0:
            raise ValueError("num_processes must be non-negative")

        self.window_size = window_size
        self.image_size = image_size
        self._epoch_size = epoch_size
        self._batch_size = int(batch_size)
        self._num_processes: Optional[int] = None if num_processes is None else int(num_processes)
        self._pre_face_transforms: Tuple[Callable[[ArrayLike], ArrayLike], ...] = ()
        self._post_face_transforms: Tuple[Callable[[ArrayLike], ArrayLike], ...] = ()
        self._pre_face_transform_funcs: Tuple[
            Callable[[ArrayLike, np.random.Generator], ArrayLike], ...
        ] = ()
        self._post_face_transform_funcs: Tuple[
            Callable[[ArrayLike, np.random.Generator], ArrayLike], ...
        ] = ()
        if pre_face_transforms is None:
            pre_face_transforms = default_pre_face_transforms(probability_scale=augment_scale)
        if post_face_transforms is None:
            post_face_transforms = default_post_face_transforms(
                window_size=self.window_size,
                probability_scale=augment_scale
            )
        self.pre_face_transforms = pre_face_transforms
        self.post_face_transforms = post_face_transforms
        self._heart_rate_as_scalar = heart_rate_as_scalar
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._cache_cameras = cache_cameras
        self._tensor_device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._tensor_dtype: torch.dtype = torch.float32
        self._camera_cache: Dict[Optional[int], Dict[str, CameraData]] = {}
        self._mp_pool: Optional[mp.pool.Pool] = None
        self._mp_pool_process_count: Optional[int] = None
        self._reset_rng_each_epoch = False
        self._timing_enabled = False
        self._timing_totals: Dict[str, float] = {}
        self._timing_counts: Dict[str, int] = {}

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
            self._shutdown_worker_pool()
        except Exception:
            pass
        try:
            self._close_camera_cache()
        except Exception:
            pass

    def __getstate__(self) -> Dict[str, object]:
        self._close_camera_cache()
        state = self.__dict__.copy()
        state["_camera_cache"] = {}
        state["_timing_enabled"] = False
        state["_timing_totals"] = {}
        state["_timing_counts"] = {}
        state["_mp_pool"] = None
        state["_mp_pool_process_count"] = None
        return state

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__dict__.update(state)
        if "_camera_cache" not in self.__dict__:
            self._camera_cache = {}
        if "_timing_enabled" not in self.__dict__:
            self._timing_enabled = False
        if "_timing_totals" not in self.__dict__:
            self._timing_totals = {}
        if "_timing_counts" not in self.__dict__:
            self._timing_counts = {}
        if "_mp_pool" not in self.__dict__:
            self._mp_pool = None
        if "_mp_pool_process_count" not in self.__dict__:
            self._mp_pool_process_count = None
        if "_reset_rng_each_epoch" not in self.__dict__:
            self._reset_rng_each_epoch = False
        if "_tensor_device" not in self.__dict__:
            self._tensor_device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        if "_tensor_dtype" not in self.__dict__:
            self._tensor_dtype = torch.float32

    # ------------------------------------------------------------------
    # Timing instrumentation
    def enable_timing(self, *, reset: bool = True) -> None:
        self._timing_enabled = True
        if reset:
            self.reset_timings()

    def disable_timing(self) -> None:
        self._timing_enabled = False

    def reset_timings(self) -> None:
        self._timing_totals.clear()
        self._timing_counts.clear()

    def collect_timings(self, *, reset: bool = False) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        for name, total in self._timing_totals.items():
            count = self._timing_counts.get(name, 0)
            avg = total / count if count else 0.0
            report[name] = {"total": total, "count": count, "avg": avg}
        if reset:
            self.reset_timings()
        return report

    def _record_timing(self, name: str, elapsed: float) -> None:
        totals = self._timing_totals
        counts = self._timing_counts
        totals[name] = totals.get(name, 0.0) + elapsed
        counts[name] = counts.get(name, 0) + 1

    @contextmanager
    def _time_section(self, name: str):
        if not self._timing_enabled:
            yield
            return
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self._record_timing(name, elapsed)

    # ------------------------------------------------------------------
    # PyTorch dataset API
    def __len__(self) -> int:
        batch_size = self._batch_size
        if batch_size <= 0:
            raise RuntimeError("Configured batch_size must be positive to determine length")
        return max(1, math.ceil(self._epoch_size / batch_size))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._time_section("__getitem__"):
            sample = self._sample_window(include_context=False)
            return self._sample_to_tuple(sample)

    # ------------------------------------------------------------------
    # Tensor configuration helpers
    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> "RandomFaceWindowDataset":
        """Adjust the device and dtype used for generated tensors."""

        if device is not None:
            self._tensor_device = torch.device(device)

        if dtype is not None:
            if isinstance(dtype, torch.dtype):
                resolved_dtype = dtype
            elif isinstance(dtype, str):
                try:
                    resolved_dtype = getattr(torch, dtype)
                except AttributeError as exc:
                    raise TypeError(f"Unsupported dtype string '{dtype}'") from exc
            else:
                raise TypeError("dtype must be a torch.dtype or string identifier")
            self._tensor_dtype = resolved_dtype

        return self

    def cpu(self) -> "RandomFaceWindowDataset":
        """Configure the dataset to emit tensors on the CPU."""

        return self.to(device=torch.device("cpu"))

    def cuda(
        self, device: Optional[Union[int, str, torch.device]] = None
    ) -> "RandomFaceWindowDataset":
        """Configure the dataset to emit tensors on a CUDA device."""

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")

        if device is None:
            target = torch.device("cuda")
        elif isinstance(device, int):
            target = torch.device("cuda", device)
        else:
            target = torch.device(device)

        if target.type != "cuda":
            target = torch.device("cuda", target.index or 0)

        return self.to(device=target)

    def float(self) -> "RandomFaceWindowDataset":
        """Configure the dataset to emit ``float32`` tensors."""

        return self.to(dtype=torch.float32)

    def double(self) -> "RandomFaceWindowDataset":
        """Configure the dataset to emit ``float64`` tensors."""

        return self.to(dtype=torch.float64)

    def get_window_with_context(
        self,
        index: int = 0,
        *,
        dataset_name: Optional[str] = None,
        video_index: Optional[int] = None,
    ) -> Dict[str, Union[str, int, torch.Tensor]]:
        """Return a sampled window that preserves context frames.

        Parameters
        ----------
        index:
            Optional deterministic index used to seed a temporary random
            generator.  Reusing the same index will yield the same sampled
            window when the dataset was initialized with a fixed seed.  The
            dataset's main random generator remains unaffected.
        dataset_name:
            Optional dataset identifier restricting the sampled window to the
            specified source dataset.  The name must match one returned by the
            ``available_datasets`` attribute.
        video_index:
            Optional index selecting a particular video within the chosen
            dataset.  When provided without ``dataset_name`` the selection will
            consider videos sharing the same index across all datasets.
        """

        with self._time_section("get_window_with_context"):
            rng = self._spawn_index_rng(index)
            return self._sample_window(
                include_context=True,
                rng=rng,
                dataset_name=dataset_name,
                video_index=video_index,
            )

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.iter_batches(include_context=False)

    def get_batch(
        self,
        *,
        include_context: bool = False,
        dataset_name: Optional[str] = None,
        video_index: Optional[int] = None,
    ) -> Union[
        Dict[str, Union[str, int, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Return a single randomized batch of windows.

        When ``include_context`` is ``False`` the method returns a tuple
        ``(frames, metadata, heart_rates)`` matching the structure produced by
        a standard PyTorch ``DataLoader`` that yields tuples.  When context is
        requested, a dictionary containing all batch fields (including
        ``"context_frames"``) is returned instead.  Multiprocessing workers are
        spawned on demand according to the dataset's configuration and are kept
        alive for reuse whenever possible.

        Parameters
        ----------
        include_context:
            When ``True`` include the per-frame context imagery in each sample.
        dataset_name, video_index:
            Optional selectors forwarded to :meth:`_sample_window` to restrict
            sampling to a particular dataset or video.
        """

        process_count = self._determine_process_count()
        pool = self._ensure_worker_pool(process_count)
        return self._sample_batch(
            include_context=include_context,
            dataset_name=dataset_name,
            video_index=video_index,
            process_count=process_count,
            pool=pool,
        )

    def iter_batches(
        self,
        *,
        include_context: bool = False,
        dataset_name: Optional[str] = None,
        video_index: Optional[int] = None,
    ) -> Iterable[
        Union[
            Dict[str, Union[str, int, torch.Tensor]],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ]:
        """Yield randomized batches using the dataset's multiprocessing setup."""

        batch_size = self._batch_size
        if batch_size <= 0:
            raise RuntimeError("Configured batch_size must be positive to sample a batch")

        rng_state = (
            copy.deepcopy(self.rng.bit_generator.state)
            if self._reset_rng_each_epoch
            else None
        )
        process_count = self._determine_process_count()
        max_batches = max(1, math.ceil(self.epoch_size / batch_size))

        pool = self._ensure_worker_pool(process_count)
        try:
            for _ in range(max_batches):
                yield self._sample_batch(
                    include_context=include_context,
                    dataset_name=dataset_name,
                    video_index=video_index,
                    process_count=process_count,
                    pool=pool,
                )
        finally:
            if rng_state is not None:
                self.rng.bit_generator.state = rng_state

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

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def tensor_device(self) -> torch.device:
        """Return the device new tensors are materialized on."""

        return self._tensor_device

    @property
    def tensor_dtype(self) -> torch.dtype:
        """Return the dtype new tensors are materialized with."""

        return self._tensor_dtype

    def set_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._batch_size = int(batch_size)

    @property
    def num_processes(self) -> Optional[int]:
        return self._num_processes

    def set_num_processes(self, num_processes: Optional[int]) -> None:
        if num_processes is None:
            self._num_processes = None
            self._shutdown_worker_pool()
            return
        value = int(num_processes)
        if value < 0:
            raise ValueError("num_processes must be non-negative")
        self._num_processes = value
        self._shutdown_worker_pool()

    def _determine_process_count(self) -> int:
        batch_size = self._batch_size
        if batch_size <= 0:
            raise RuntimeError("Configured batch_size must be positive to sample a batch")

        configured_processes = self._num_processes
        if configured_processes is None:
            cpu_count = os.cpu_count() or 1
            return max(1, min(batch_size, cpu_count))

        return max(1, min(batch_size, int(configured_processes)))

    def _sample_batch(
        self,
        *,
        include_context: bool,
        dataset_name: Optional[str],
        video_index: Optional[int],
        process_count: Optional[int] = None,
        pool: Optional[mp.pool.Pool] = None,
    ) -> Union[
        Dict[str, Union[str, int, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        batch_size = self._batch_size
        if batch_size <= 0:
            raise RuntimeError("Configured batch_size must be positive to sample a batch")

        if process_count is None:
            process_count = self._determine_process_count()

        if process_count <= 1 or pool is None:
            samples = [
                self._sample_window(
                    include_context=include_context,
                    dataset_name=dataset_name,
                    video_index=video_index,
                )
                for _ in range(batch_size)
            ]
        else:
            max_seed = np.iinfo(np.uint64).max
            task_args = [
                (
                    include_context,
                    dataset_name,
                    video_index,
                    int(self.rng.integers(0, max_seed, dtype=np.uint64)),
                )
                for _ in range(batch_size)
            ]
            samples = pool.map(_multiprocess_worker_sample, task_args)

        collated = default_collate(samples)
        if include_context:
            return collated
        return self._collated_to_tuple(collated)

    @staticmethod
    def _sample_to_tuple(
        sample: Dict[str, Union[str, int, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frames = sample["frames"]
        metadata = sample["face_metadata"]
        heart_rates = sample["heart_rates"]
        if not isinstance(frames, torch.Tensor) or not isinstance(
            metadata, torch.Tensor
        ) or not isinstance(heart_rates, torch.Tensor):
            raise TypeError("Sample components must be torch.Tensors")
        return frames, metadata, heart_rates

    def _collated_to_tuple(
        self, collated: Dict[str, Union[str, int, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frames = collated["frames"]
        metadata = collated["face_metadata"]
        heart_rates = collated["heart_rates"]
        if not isinstance(frames, torch.Tensor) or not isinstance(
            metadata, torch.Tensor
        ) or not isinstance(heart_rates, torch.Tensor):
            raise TypeError("Batch components must be torch.Tensors")
        return frames, metadata, heart_rates

    def _ensure_worker_pool(self, process_count: int) -> Optional[mp.pool.Pool]:
        if process_count <= 1:
            self._shutdown_worker_pool()
            return None

        pool = self._mp_pool
        if pool is not None and self._mp_pool_process_count == process_count:
            return pool

        self._shutdown_worker_pool()
        ctx = _resolve_multiprocessing_context()
        state = self.__getstate__()

        pool = ctx.Pool(
            processes=process_count,
            initializer=_multiprocess_worker_init,
            initargs=(state,),
        )
        self._mp_pool = pool
        self._mp_pool_process_count = process_count
        return pool

    def _shutdown_worker_pool(self) -> None:
        pool = getattr(self, "_mp_pool", None)
        if pool is not None:
            try:
                pool.close()
            finally:
                pool.join()
        self._mp_pool = None
        self._mp_pool_process_count = None

    def train_test_split(
        self,
        train_proportion: float,
        *,
        seed: Optional[int] = None,
    ) -> Tuple["RandomFaceWindowDataset", "RandomFaceWindowDataset"]:
        """Split the dataset into training and testing subsets per video.

        Parameters
        ----------
        train_proportion:
            Proportion of videos assigned to the training subset.  Must be in the
            open interval ``(0.0, 1.0)``.
        seed:
            Optional random seed controlling the shuffling applied before
            splitting.  When omitted the dataset's own seed is reused.

        Returns
        -------
        Tuple[RandomFaceWindowDataset, RandomFaceWindowDataset]
            A pair of datasets containing disjoint sets of videos.
        """

        with self._time_section("train_test_split"):
            if not 0.0 < train_proportion < 1.0:
                raise ValueError("train_proportion must be between 0 and 1")

            if len(self._videos) < 2:
                raise ValueError("At least two videos are required to perform a split")

            base_seed = seed if seed is not None else self.seed
            rng = np.random.default_rng(base_seed)

            indices = np.arange(len(self._videos))
            rng.shuffle(indices)

            split_idx = int(round(len(indices) * train_proportion))
            split_idx = max(1, min(len(indices) - 1, split_idx))

            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]

            train_videos = [self._videos[int(idx)] for idx in train_indices]
            test_videos = [self._videos[int(idx)] for idx in test_indices]

            train_seed = None if base_seed is None else int(base_seed) + 1
            test_seed = None if base_seed is None else int(base_seed) + 2

            train_dataset = self._clone_with_videos(
                train_videos,
                seed=train_seed,
                reset_rng_each_epoch=False,
            )
            test_dataset = self._clone_with_videos(
                test_videos,
                seed=test_seed,
                reset_rng_each_epoch=True,
            )

            return train_dataset, test_dataset

    # ------------------------------------------------------------------
    # Internal helpers
    def _get_rng(self) -> np.random.Generator:
        with self._time_section("_get_rng"):
            return self.rng

    def _clone_with_videos(
        self,
        videos: Sequence[_VideoEntry],
        *,
        seed: Optional[int],
        reset_rng_each_epoch: Optional[bool] = None,
    ) -> "RandomFaceWindowDataset":
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__ = self.__dict__.copy()
        clone._videos = list(videos)
        clone._camera_cache = {}
        clone.seed = seed
        clone.rng = np.random.default_rng(seed)
        clone._timing_enabled = False
        clone._timing_totals = {}
        clone._timing_counts = {}
        clone._mp_pool = None
        clone._mp_pool_process_count = None
        if reset_rng_each_epoch is None:
            clone._reset_rng_each_epoch = getattr(self, "_reset_rng_each_epoch", False)
        else:
            clone._reset_rng_each_epoch = bool(reset_rng_each_epoch)
        return clone

    def _spawn_index_rng(self, index: int) -> np.random.Generator:
        with self._time_section("_spawn_index_rng"):
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
        dataset_name: Optional[str] = None,
        video_index: Optional[int] = None,
    ) -> Dict[str, Union[str, int, torch.Tensor]]:
        with self._time_section("_sample_window"):
            active_rng = self._get_rng() if rng is None else rng
            entry, start_frame = self._select_entry_and_start(
                active_rng,
                dataset_name=dataset_name,
                video_index=video_index,
            )
            self._reset_transform_state(active_rng)
            frames, face_metadata, context_frames = self._read_face_window(
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
                assert context_frames is not None
                sample["context_frames"] = context_frames
            return sample

    def _select_entry_and_start(
        self,
        rng: np.random.Generator,
        *,
        dataset_name: Optional[str] = None,
        video_index: Optional[int] = None,
    ) -> Tuple[_VideoEntry, int]:
        with self._time_section("_select_entry_and_start"):
            candidates = self._videos
            if dataset_name is not None:
                candidates = [
                    entry for entry in candidates if entry.dataset_name == dataset_name
                ]
                if not candidates:
                    raise ValueError(
                        f"No videos available for dataset '{dataset_name}'"
                    )

            if video_index is not None:
                candidates = [
                    entry for entry in candidates if entry.dataset_index == video_index
                ]
                if not candidates:
                    raise ValueError(
                        "No videos available matching the requested dataset/video index"
                    )

            entry_idx = int(rng.integers(0, len(candidates)))
            entry = candidates[entry_idx]
            max_start = entry.num_frames - self.window_size
            if max_start <= 0:
                raise RuntimeError(
                    f"Video '{entry.video_path}' does not contain enough frames for a window"
                )
            start_frame = int(rng.integers(0, max_start + 1))
            return entry, start_frame

    def _reset_transform_state(self, rng: np.random.Generator) -> None:
        with self._time_section("_reset_transform_state"):
            for transform in self._pre_face_transforms + self._post_face_transforms:
                self._invoke_transform_reset(transform, rng)

    def _invoke_transform_reset(
        self, transform: Callable[[ArrayLike], ArrayLike], rng: np.random.Generator
    ) -> None:
        with self._time_section("_invoke_transform_reset"):
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
        with self._time_section("_get_cached_camera"):
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
        with self._time_section("_invalidate_cached_camera"):
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
        with self._time_section("_prepare_video_entries"):
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
        with self._time_section("_build_video_entry"):
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
        with self._time_section("_slice_heart_rates"):
            end_frame = start_frame + self.window_size
            heart_rates = entry.heart_rates[start_frame:end_frame]
            if heart_rates.shape[0] != self.window_size:
                raise RuntimeError(
                    "Heart rate annotations do not cover the requested window"
                )
            values = torch.as_tensor(heart_rates, dtype=torch.float32)
            if self._heart_rate_as_scalar:
                if values.numel() == 0:
                    raise RuntimeError("No heart-rate values available for the requested window")
                return self._convert_tensor(values.mean())
            return self._convert_tensor(values)

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
        with self._time_section("_read_face_window"):
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
                    (self.window_size, 6, self.image_size, self.image_size),
                    dtype=self._tensor_dtype,
                    device=self._tensor_device,
                )
                metadata = torch.empty(
                    (self.window_size, 5),
                    dtype=self._tensor_dtype,
                    device=self._tensor_device,
                )
                context_frames_list: List[torch.Tensor] = []
                last_bbox: Optional[Tuple[int, int, int, int]] = None
                prev_rgb_view: Optional[torch.Tensor] = None

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
                    face_tensor = self._convert_tensor(face_tensor)
                    rgb_view = face_frames[frame_idx, :3]
                    rgb_view.copy_(face_tensor)
                    diff_view = face_frames[frame_idx, 3:]
                    if prev_rgb_view is None:
                        diff_view.zero_()
                    else:
                        torch.sub(rgb_view, prev_rgb_view, out=diff_view)
                    prev_rgb_view = rgb_view
                    if include_context:
                        context_frames_list.append(self._prepare_context_frame(frame_bgr))
                    combined_visibility = visibility * post_visibility
                    metadata[frame_idx].copy_(
                        self._build_face_metadata(
                            adjusted_bbox,
                            combined_visibility,
                            frame_bgr.shape,
                        )
                    )

                if include_context:
                    if not context_frames_list:
                        raise RuntimeError("Context frames were requested but none were captured")
                    context_frames = torch.stack(context_frames_list, dim=0)
                else:
                    context_frames = None

                return face_frames, metadata, context_frames
            finally:
                if owns_camera:
                    camera.close()

    def _fast_forward_camera(self, camera: CameraData, frames_to_skip: int) -> None:
        with self._time_section("_fast_forward_camera"):
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
        with self._time_section("_apply_pre_transforms"):
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
        with self._time_section("_apply_post_transforms"):
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

    def _prepare_context_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        with self._time_section("_prepare_context_frame"):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            context_tensor = torch.as_tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1)
            if context_tensor.max() > 1.0:
                context_tensor.mul_(1.0 / 255.0)
            return self._convert_tensor(context_tensor)

    def _apply_transforms(
        self,
        transforms: Sequence[Callable[[ArrayLike, np.random.Generator], ArrayLike]],
        data: ArrayLike,
        rng: np.random.Generator,
    ) -> ArrayLike:
        with self._time_section("_apply_transforms"):
            result: ArrayLike = data
            for transform in transforms:
                result = transform(result, rng)
            return result

    def _detect_face(
        self,
        frame_bgr: np.ndarray,
        previous_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[Tuple[int, int, int, int], float]:
        with self._time_section("_detect_face"):
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
        with self._time_section("_extract_face"):
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
        with self._time_section("_build_face_metadata"):
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
            return self._convert_tensor(metadata)

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

    def _convert_tensor(
        self,
        tensor: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        target_dtype = self._tensor_dtype if dtype is None else dtype
        target_device = self._tensor_device if device is None else device
        if tensor.dtype != target_dtype or tensor.device != target_device:
            try:
                tensor = tensor.to(device=target_device, dtype=target_dtype, copy=False)
            except TypeError:
                tensor = tensor.to(device=target_device, dtype=target_dtype)
        return tensor

    def _to_normalized_tensor(self, face_like: ArrayLike) -> torch.Tensor:
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
            return self._convert_tensor(tensor)

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
        return self._convert_tensor(tensor)
def _resolve_multiprocessing_context() -> mp.context.BaseContext:
    available = mp.get_all_start_methods()
    method = "fork" if "fork" in available else "spawn"
    return mp.get_context(method)


_MULTIPROCESS_DATASET: Optional["RandomFaceWindowDataset"] = None


def _multiprocess_worker_init(state: Dict[str, object]) -> None:
    global _MULTIPROCESS_DATASET

    dataset = RandomFaceWindowDataset.__new__(RandomFaceWindowDataset)
    dataset.__setstate__(state)
    base_seed = getattr(dataset, "seed", None)
    dataset.rng = np.random.default_rng(base_seed)

    _MULTIPROCESS_DATASET = dataset


def _multiprocess_worker_sample(
    args: Tuple[bool, Optional[str], Optional[int], Optional[int]]
) -> Dict[str, Union[str, int, torch.Tensor]]:
    include_context, dataset_name, video_index, seed = args

    if _MULTIPROCESS_DATASET is None:
        raise RuntimeError("Multiprocessing dataset worker not initialized")

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    return _MULTIPROCESS_DATASET._sample_window(
        include_context=include_context,
        dataset_name=dataset_name,
        video_index=video_index,
        rng=rng,
    )
__all__ = ["RandomFaceWindowDataset"]

