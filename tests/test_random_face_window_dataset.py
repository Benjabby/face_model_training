import numpy as np
import pytest
import torch

from face_model_training.datasets.face_window_dataset import RandomFaceWindowDataset, _VideoEntry


def _make_dataset_stub() -> RandomFaceWindowDataset:
    dataset = RandomFaceWindowDataset.__new__(RandomFaceWindowDataset)
    dataset.window_size = 3
    dataset.image_size = 2
    dataset._epoch_size = 1
    dataset._batch_size = 1
    dataset._num_processes = None
    dataset._pre_face_transforms = ()
    dataset._post_face_transforms = ()
    dataset._pre_face_transform_funcs = ()
    dataset._post_face_transform_funcs = ()
    dataset._heart_rate_as_scalar = False
    dataset.seed = None
    dataset.rng = np.random.default_rng()
    dataset._cache_cameras = False
    dataset._tensor_device = torch.device("cpu")
    dataset._tensor_dtype = torch.float32
    dataset._camera_cache = {}
    dataset._mp_pool = None
    dataset._mp_pool_process_count = None
    dataset._reset_rng_each_epoch = False
    dataset._timing_enabled = False
    dataset._timing_totals = {}
    dataset._timing_counts = {}
    dataset._videos = []
    return dataset


def test_dtype_switch_affects_metadata() -> None:
    dataset = _make_dataset_stub()

    meta = dataset._build_face_metadata((0, 0, 1, 1), 0.5, (10, 10, 3))
    assert meta.dtype == torch.float32

    dataset.double()
    meta64 = dataset._build_face_metadata((0, 0, 1, 1), 0.5, (10, 10, 3))
    assert meta64.dtype == torch.float64


def test_slice_heart_rates_respects_device_and_dtype() -> None:
    dataset = _make_dataset_stub()
    rates = np.linspace(60.0, 62.0, dataset.window_size, dtype=np.float32)
    entry = _VideoEntry(
        dataset_name="stub",
        dataset_index=0,
        video_path="stub",
        num_frames=dataset.window_size,
        heart_rates=rates,
        frame_times=np.linspace(0.0, 1.0, dataset.window_size, dtype=np.float32),
    )

    tensor = dataset._slice_heart_rates(entry, 0)
    assert tensor.device == dataset._tensor_device
    assert tensor.dtype == dataset._tensor_dtype

    dataset._heart_rate_as_scalar = True
    scalar = dataset._slice_heart_rates(entry, 0)
    assert scalar.device == dataset._tensor_device
    assert scalar.dtype == dataset._tensor_dtype


def test_prepare_context_frame_uses_target_options() -> None:
    dataset = _make_dataset_stub()
    dataset.double()
    frame = np.full((2, 2, 3), 128, dtype=np.uint8)

    context = dataset._prepare_context_frame(frame)
    assert context.device == dataset._tensor_device
    assert context.dtype == dataset._tensor_dtype
    assert context.dtype == torch.float64


class _DummyCamera:
    def __init__(self, frames: np.ndarray) -> None:
        self._frames = frames
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._frames):
            raise StopIteration
        frame = self._frames[self._index]
        time = float(self._index) / 30.0
        self._index += 1
        return frame, time

    def close(self) -> None:  # pragma: no cover - not used but required by interface
        pass

    def reset(self) -> None:  # pragma: no cover - simple stub
        self._index = 0

    def skip(self, frames_to_skip: int) -> None:
        self._index += frames_to_skip


def test_read_face_window_produces_expected_channels() -> None:
    dataset = _make_dataset_stub()
    dataset._apply_pre_transforms = lambda frame, rng: frame
    dataset._detect_face = lambda frame, last_bbox: (
        (0, 0, frame.shape[1], frame.shape[0]),
        1.0,
    )
    dataset._extract_face = lambda frame, bbox: (frame, bbox)

    def _post(face_rgb, rng):
        tensor = torch.as_tensor(face_rgb, dtype=torch.float32).permute(2, 0, 1)
        tensor.mul_(1.0 / 255.0)
        return tensor, 1.0

    dataset._apply_post_transforms = _post  # type: ignore[assignment]

    frames = np.stack(
        [
            np.full((dataset.image_size, dataset.image_size, 3), fill_value=40 * (i + 1), dtype=np.uint8)
            for i in range(dataset.window_size)
        ],
        axis=0,
    )
    camera = _DummyCamera(frames)
    entry = _VideoEntry(
        dataset_name="stub",
        dataset_index=0,
        video_path="stub",
        num_frames=len(frames),
        heart_rates=np.linspace(60.0, 64.0, len(frames), dtype=np.float32),
        frame_times=np.linspace(0.0, 1.0, len(frames), dtype=np.float32),
    )

    window, metadata, context = dataset._read_face_window(
        entry,
        start_frame=0,
        rng=np.random.default_rng(),
        include_context=False,
        camera=camera,
    )

    assert window.shape == (dataset.window_size, 6, dataset.image_size, dataset.image_size)
    assert torch.allclose(window[0, 3:], torch.zeros_like(window[0, 3:]))
    expected_diff = window[1, :3] - window[0, :3]
    assert torch.allclose(window[1, 3:], expected_diff)
    assert metadata.shape == (dataset.window_size, 5)
    assert context is None


def test_cuda_switch_matches_availability() -> None:
    dataset = _make_dataset_stub()
    if torch.cuda.is_available():
        dataset.cuda()
        assert dataset._tensor_device.type == "cuda"
    else:
        with pytest.raises(RuntimeError):
            dataset.cuda()
