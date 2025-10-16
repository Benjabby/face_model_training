"""Common augmentation utilities for :mod:`datasets.face_window_dataset`."""

from __future__ import annotations

from typing import Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except Exception:  # pragma: no cover - torch may not be available at import time
    torch = None  # type: ignore[assignment]

ArrayLike = Union[np.ndarray, "torch.Tensor", Image.Image]


def _to_numpy(array_like: ArrayLike) -> np.ndarray:
    if torch is not None and isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    if isinstance(array_like, Image.Image):
        return np.array(array_like)
    return np.asarray(array_like)


def _ensure_uint8(array_like: ArrayLike) -> np.ndarray:
    array = _to_numpy(array_like)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return array


class RandomHorizontalFlip:
    """Flips the entire window horizontally based on a per-window decision."""

    def __init__(self, prob: float = 0.5) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must lie in [0, 1]")
        self.prob = prob
        self._enabled: bool = False

    def reset(self, window_size: int, rng: np.random.Generator) -> None:  # pragma: no cover - simple setter
        self._enabled = bool(rng.random() < self.prob)

    def __call__(self, frame: ArrayLike, rng: np.random.Generator) -> np.ndarray:
        frame_uint8 = _ensure_uint8(frame)
        if not self._enabled:
            return frame_uint8
        return np.ascontiguousarray(frame_uint8[:, ::-1])


class RandomGaussianNoise:
    """Injects low-amplitude Gaussian noise with parameters stable per window."""

    def __init__(self, sigma_range: Tuple[float, float] = (2.0, 6.0), prob: float = 0.5) -> None:
        self.sigma_range = sigma_range
        self.prob = prob
        self._sigma: float = sigma_range[0]
        self._enabled: bool = False

    def reset(self, window_size: int, rng: np.random.Generator) -> None:  # pragma: no cover - simple setter
        self._enabled = bool(rng.random() < self.prob)
        if self._enabled:
            self._sigma = float(rng.uniform(*self.sigma_range))

    def __call__(self, frame: ArrayLike, rng: np.random.Generator) -> np.ndarray:
        if not self._enabled:
            return _ensure_uint8(frame)
        frame_uint8 = _ensure_uint8(frame)
        noise = rng.normal(0.0, self._sigma, size=frame_uint8.shape)
        noisy = frame_uint8.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy


class RandomColorJitter:
    """Applies gentle brightness and contrast adjustments per window."""

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.05,
        prob: float = 0.7,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.prob = prob
        self._alpha: float = 1.0
        self._beta: float = 0.0
        self._saturation_scale: float = 1.0
        self._enabled: bool = False

    def reset(self, window_size: int, rng: np.random.Generator) -> None:  # pragma: no cover - simple setter
        self._enabled = bool(rng.random() < self.prob)
        if not self._enabled:
            return
        self._alpha = float(rng.uniform(1.0 - self.contrast, 1.0 + self.contrast))
        self._beta = float(rng.uniform(-255.0 * self.brightness, 255.0 * self.brightness))
        self._saturation_scale = float(rng.uniform(1.0 - self.saturation, 1.0 + self.saturation))

    def __call__(self, frame: ArrayLike, rng: np.random.Generator) -> np.ndarray:
        frame_uint8 = _ensure_uint8(frame)
        if not self._enabled:
            return frame_uint8

        adjusted = cv2.convertScaleAbs(frame_uint8, alpha=self._alpha, beta=self._beta)
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[..., 1] *= self._saturation_scale
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        jittered = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return jittered


class RandomSmallAffine:
    """Applies small translations and rotations that are stable per window."""

    def __init__(
        self,
        max_rotation: float = 3.0,
        max_translation: float = 0.02,
        prob: float = 0.5,
    ) -> None:
        self.max_rotation = max_rotation
        self.max_translation = max_translation
        self.prob = prob
        self._matrix: Optional[np.ndarray] = None

    def reset(self, window_size: int, rng: np.random.Generator) -> None:  # pragma: no cover - simple setter
        if rng.random() < self.prob:
            angle = rng.uniform(-self.max_rotation, self.max_rotation)
            tx = rng.uniform(-self.max_translation, self.max_translation)
            ty = rng.uniform(-self.max_translation, self.max_translation)
            self._matrix = (angle, tx, ty)
        else:
            self._matrix = None

    def __call__(self, frame: ArrayLike, rng: np.random.Generator) -> np.ndarray:
        frame_uint8 = _ensure_uint8(frame)
        if self._matrix is None:
            return frame_uint8
        angle, tx, ty = self._matrix
        h, w = frame_uint8.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        matrix[0, 2] += tx * w
        matrix[1, 2] += ty * h
        warped = cv2.warpAffine(
            frame_uint8,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return warped


class RandomWindowAffine:
    """Applies moderate affine transforms consistently across a window."""

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-5.0, 5.0),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        shear_range: Tuple[float, float] = (-0.03, 0.03),
        translation_range: Tuple[float, float] = (-0.03, 0.03),
        prob: float = 0.35,
    ) -> None:
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.translation_range = translation_range
        self.prob = prob
        self._params: Optional[Tuple[float, ...]] = None

    def reset(self, window_size: int, rng: np.random.Generator) -> None:  # pragma: no cover - simple setter
        if rng.random() >= self.prob:
            self._params = None
            return

        angle = float(rng.uniform(*self.rotation_range))
        scale_x = float(rng.uniform(*self.scale_range))
        scale_y = float(rng.uniform(*self.scale_range))
        shear_x = float(rng.uniform(*self.shear_range))
        shear_y = float(rng.uniform(*self.shear_range))
        trans_x = float(rng.uniform(*self.translation_range))
        trans_y = float(rng.uniform(*self.translation_range))
        self._params = (angle, scale_x, scale_y, shear_x, shear_y, trans_x, trans_y)

    def __call__(self, frame: ArrayLike, rng: np.random.Generator) -> np.ndarray:
        frame_uint8 = _ensure_uint8(frame)
        if self._params is None:
            return frame_uint8

        angle, scale_x, scale_y, shear_x, shear_y, trans_x, trans_y = self._params
        h, w = frame_uint8.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        rad = np.deg2rad(angle)
        cos_a = float(np.cos(rad))
        sin_a = float(np.sin(rad))

        rotation = np.array(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        scale = np.array(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        shear = np.array(
            [[1.0, shear_x, 0.0], [shear_y, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        translation = np.array(
            [[1.0, 0.0, trans_x * w], [0.0, 1.0, trans_y * h], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        to_center = np.array(
            [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        from_center = np.array(
            [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        matrix = from_center @ translation @ shear @ scale @ rotation @ to_center
        affine = matrix[:2]

        warped = cv2.warpAffine(
            frame_uint8,
            affine,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return warped


class RandomFrameBlackout:
    """Randomly zeroes frames while respecting a minimum mean visibility constraint."""

    def __init__(
        self,
        blackout_probability: float = 0.15,
        min_mean_visibility: float = 0.6,
    ) -> None:
        if not 0.0 <= min_mean_visibility <= 1.0:
            raise ValueError("min_mean_visibility must lie in [0, 1]")
        self.blackout_probability = blackout_probability
        self.min_mean_visibility = min_mean_visibility
        self._indices: Set[int] = set()
        self._cursor: int = 0
        self._window_size: int = 0
        self._visibility_floor: float = 1.0

    def reset(self, window_size: int, rng: np.random.Generator) -> None:  # pragma: no cover - simple setter
        self._window_size = window_size
        self._cursor = 0
        if window_size <= 0:
            self._indices = set()
            return
        high = np.nextafter(1.0, np.inf)
        self._visibility_floor = float(
            rng.uniform(self.min_mean_visibility, high)
        )
        max_blackouts = max(0, int(np.floor((1.0 - self._visibility_floor) * window_size)))
        if max_blackouts == 0:
            self._indices = set()
            return
        desired = int(rng.binomial(window_size, self.blackout_probability))
        count = min(desired, max_blackouts)
        if count <= 0:
            self._indices = set()
            return
        choices = rng.choice(window_size, size=count, replace=False)
        self._indices = set(int(i) for i in np.asarray(choices, dtype=int))

    def __call__(self, frame: ArrayLike, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
        frame_uint8 = _ensure_uint8(frame)
        index = self._cursor
        self._cursor += 1
        if index in self._indices:
            return np.zeros_like(frame_uint8), 0.0
        return frame_uint8, 1.0


def default_pre_face_transforms() -> Sequence[object]:
    """Return a set of gentle default pre-face transforms."""

    return (
        RandomHorizontalFlip(),
        RandomGaussianNoise(),
        RandomColorJitter(),
        RandomSmallAffine(),
        RandomWindowAffine(),
    )


def default_post_face_transforms(
    window_size: int,
    blackout_probability: float = 0.15,
    min_mean_visibility: float = 0.6,
) -> Sequence[object]:
    """Return default post-face transforms, including randomized blackouts."""

    blackout = RandomFrameBlackout(
        blackout_probability=blackout_probability,
        min_mean_visibility=min_mean_visibility,
    )
    blackout.reset(window_size=window_size, rng=np.random.default_rng())
    return (blackout,)


__all__ = [
    "RandomHorizontalFlip",
    "RandomGaussianNoise",
    "RandomColorJitter",
    "RandomSmallAffine",
    "RandomWindowAffine",
    "RandomFrameBlackout",
    "default_pre_face_transforms",
    "default_post_face_transforms",
]
