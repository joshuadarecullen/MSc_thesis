import numpy as np
from numpy.typing import NDArray
from typing import Callable

def frame_to_seconds(
    frame_length: int,
    sample_rate: int,
) -> float:
    return frame_length / sample_rate

def seconds_to_frame(
    duration_seconds: int,
    sample_rate: int,
) -> int:
    return int(round(sample_rate * duration_seconds))

def fft_length(window_length: int):
    return 2 ** int(np.ceil(np.log(window_length) / np.log(2.0)))

def hertz_to_mel(
    frequencies_hz: NDArray,
    scaling_factor: float = 1127.0,
    break_frequency_hz: float = 700.0,
    log: Callable = np.log,
) -> NDArray:
    return scaling_factor * log(1 + (frequencies_hz / break_frequency_hz))

def stft_hop_length_seconds_to_feature_sample_rate(stft_hop_length_seconds: float) -> float:
    return 1.0 / stft_hop_length_seconds

