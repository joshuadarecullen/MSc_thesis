import librosa
import torch
import numpy as np
from numpy.typing import NDArray
from torch.nn import Module, Sequential
from torch import Tensor
from typing import (
    Callable,
    Optional,
    Tuple,
    Union,
)

from .utils import (
    frame_to_seconds,
    seconds_to_frame,
    fft_length,
    hertz_to_mel,
)

__all__ = ["Spectrogram"]

"""
convert a time series signal into a spectrogram using scipy's implementation of the
fast fourier transform (FFT) algorithm which performs a discrete fourier transform (DFT)
to decompose sine waves within frequency bands determined by the size of the FFT window

adds additional functionality to perform a mel (logarithmic) scaling of the spectrogram,
whereby we adjust the frequency spacing according to human perceptual scale, i.e. according
to equally spaced 'pitches'. magnitude scaling can also be applied to perform a logarithmic
transformation of the magnitude, converting magnitude to dB scale (where dB is relative to
peak power)

implementation notes:
- 06.2022 - implemented according to parameter requirements to construct spectrogram according
            to VGGish's specifications. largely copied code from their pytorch implementation
- 08.2022 - adjusted to use librosa to simplify the code dramatically
- 11.2022 - librosa prevented us from setting custom values for the scaling coefficient and
            break frequency in the HTK algorithm (mapping hz to mel) and the logarithmic base
            used to space the mel frequency bins. adjusting these was a requirement to construct
            birdnet's representation.
- 11.2022 - reverted to using librosa for the DFT, since scipy was behaving differently to as
            expected. frequency range for spectrogram is computed manually between, i.e. 0 - nyquist
          - ensure spectrogram returns time on the x-axis for easy use with frame, etc
- 02.2023 - librosa by default uses the slaney formula (i.e. linear up to 1000 hz), whereas htk
            is logarithmic from 0 hz. librosa does not allow for setting different break frequency
            and scaling factors for the mel htk formula, so we've reverted to the original VGGish code
            including our adaptations using librosa for the stft and bird net's optional db magnitude scale

development notes:
- 11.2022 - we don't need the change of log base, just solve mel equation to calculate correct HTK constants
          - enable normalisation w.r.t. global dB scale rather than relative to single spectrogram
          - cache unique phase following FFT to enable access in backward function
          - implement backward using griffinlim algorithm to estimate phase and allow for options
          - implement new option using a trained GAN to estimate phase
"""


class Spectrogram(Module):
    def __init__(
        self,
        sample_rate: int,
        window_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        stft_window_length_seconds: Optional[float] = None,
        stft_hop_length_seconds: Optional[float] = None,
        overlap: Optional[Union[float, int]] = None,
        return_phase: bool = False,
        frequency_scale: str = "linear",
        num_mel_bins: Optional[int] = 64,
        min_hz: Optional[float] = 125.0,
        max_hz: Optional[float] = 7500.0,
        mel_scaling_factor: float = 1127.0,
        mel_break_frequency_hz: float = 700.0,
        hz_to_mel_base: float = np.e,
        magnitude_scale: Optional[str] = None,
        top_db: Optional[float] = 100.0,
        log_offset: Optional[float] = 1e-3,
    ) -> None:
        super().__init__()
        if return_phase:
            assert frequency_scale != "mel", \
                "cannot return the phase with a mel spectrogram"
        self.sample_rate = sample_rate
        # fft params
        self._calculate_window(window_length, stft_window_length_seconds)
        self._calculate_hop(hop_length, stft_hop_length_seconds, overlap)
        self.fft_length = fft_length(self.window_length)
        # bandpass and mel min/max filter params
        self.min_hz = min_hz
        self.max_hz = max_hz
        # frequency params
        self.frequency_scale = frequency_scale
        # mel params
        self.num_mel_bins = num_mel_bins
        self.hz_to_mel_base = hz_to_mel_base
        self.mel_scaling_factor = mel_scaling_factor
        self.mel_break_frequency_hz = mel_break_frequency_hz
        # magnitude params
        self.magnitude_scale = magnitude_scale
        self.top_db = top_db
        self.log_offset = log_offset
        self.return_phase = return_phase

    @property
    def nyquist_hz(self):
        return self.sample_rate / 2.0

    def forward(self, data: Tensor) -> Tensor:
        signal = data.flatten().numpy()
        # compute discrete fourier transform using librosa's fast fourier transform
        fft = librosa.stft(
            signal,
            win_length=self.window_length,
            hop_length=self.hop_length,
            n_fft=self.fft_length,
            window="hann",
        )
        # extract the magnitude, discard phase
        magnitude, phase = librosa.magphase(fft)
        # if we want the spectrogram in the mel scale, i.e. adjusted according to
        # perceptual scale for equal spacing of pitches, we apply a logarithmic
        # transformation of spectrogram frequency bins using a mel filterbank matrix
        num_spectrogram_bins = magnitude.shape[0]
        if self.frequency_scale == "mel":
            mel_filterbank = self.mel_filterbank(num_spectrogram_bins)
            mel_spectrogram = (magnitude.transpose(1, 0) @ mel_filterbank).transpose(1, 0)
            magnitude = mel_spectrogram
        # apply magnitude scaling
        if self.magnitude_scale == "db":
            # shift from magnitude to power spectrum
            magnitude = magnitude ** 2
            # convert power spectrum to dB scale (compute dB relative to peak power)
            magnitude = 10.0 * np.log10(np.maximum(self.log_offset, magnitude))
            magnitude = np.maximum(magnitude, magnitude.max() - self.top_db)
        # for log mel spectrograms
        elif self.magnitude_scale == "log":
            magnitude= np.log(np.maximum(self.log_offset, magnitude))
        # remap to tensor on device with added channel dimension
        magnitude = torch.as_tensor(magnitude.transpose(1, 0), device=data.device).unsqueeze(0)
        # return the phase along second channel if requested
        if self.return_phase and self.frequency_scale != "mel":
            phase = torch.as_tensor(np.angle(phase).transpose(1, 0), device=data.device).unsqueeze(0)
            return torch.cat([magnitude, phase], dim=0)
        # otherwise we just return the magnitude
        return magnitude

    def mel_filterbank(self, num_spectrogram_bins: int):
        spectrogram_bins_hertz = np.linspace(0.0, self.nyquist_hz, num_spectrogram_bins)
        min_mel, max_mel = hertz_to_mel(
          np.array([self.min_hz, self.max_hz]),
          scaling_factor=self.mel_scaling_factor,
          break_frequency_hz=self.mel_break_frequency_hz,
          log=lambda x: np.log(x) / np.log(self.hz_to_mel_base),
        )
        spectrogram_bins_mel = hertz_to_mel(
          spectrogram_bins_hertz,
          scaling_factor=self.mel_scaling_factor,
          break_frequency_hz=self.mel_break_frequency_hz,
          log=lambda x: np.log(x) / np.log(self.hz_to_mel_base),
        )
        # The i'th mel band (starting from i=1) has center frequency
        # mel_bands[i], lower edge mel_bands[i-1], and higher edge
        # mel_bands[i+1].  Thus, we need num_mel_bins + 2 values in
        # the mel_bands arrays.
        mel_bands = np.linspace(min_mel, max_mel, self.num_mel_bins + 2)
        filter_bank = np.empty((num_spectrogram_bins, self.num_mel_bins))
        for i in range(self.num_mel_bins):
            lower_edge_mel, center_mel, upper_edge_mel = mel_bands[i:i + 3]
            # calculate lower and upper slopes for every spectrogram bin.
            # line segments are linear in the *mel* domain, not hertz.
            lower_slope = ((spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel))
            upper_slope = ((upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel))
            # then intersect them with each other and zero.
            filter_bank[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
        # HTK excludes the spectrogram DC bin; make sure it always gets a zero coefficient.
        filter_bank[0, :] = 0.0
        return filter_bank

    def _calculate_window(
        self,
        window_length: Optional[int],
        stft_window_length_seconds: Optional[float],
    ) -> None:
        if stft_window_length_seconds is not None:
            assert isinstance(stft_window_length_seconds, float), "'stft_window_length_seconds' must be a float"
            self.stft_window_length_seconds = stft_window_length_seconds
            self.window_length = seconds_to_frame(stft_window_length_seconds, self.sample_rate)
        elif window_length is not None:
            assert isinstance(window_length, int), "'window_length' must be an integer"
            self.stft_window_length_seconds = frame_to_seconds(window_length, self.sample_rate)
            self.window_length = window_length
        else:
            raise ValueError("size or duration of window must be specified")

    def _calculate_hop(
        self,
        hop_length: Optional[int],
        stft_hop_length_seconds: Optional[float],
        overlap: Optional[Union[float, int]],
    ) -> None:
        if overlap is not None:
            if isinstance(overlap, float):
                assert 0 <= overlap <= 1, "'overlap' must be between 0 and 1"
                self.stft_hop_length_seconds = (
                    self.stft_window_length_seconds - (self.stft_window_length_seconds * overlap)
                )
                self.overlap = int(self.window_length * overlap)
                self.hop_length = self.window_length - self.overlap
            elif isinstance(overlap, int):
                assert overlap <= self.window_length, "'hop length' greater than window length not supported"
                self.hop_length = self.window_length - overlap
                self.stft_hop_length_seconds = frame_to_seconds(self.hop_length, self.sample_rate)
                self.overlap = overlap
            else:
                raise ValueError("'overlap' must be a float or integer")
        else:
            if stft_hop_length_seconds is not None:
                self.stft_hop_length_seconds = stft_hop_length_seconds
                self.hop_length = seconds_to_frame(stft_hop_length_seconds, self.sample_rate)
            elif hop_length is not None:
                self.stft_hop_length_seconds = frame_to_seconds(hop_length, self.sample_rate)
                self.hop_length = hop_length
            else:
                raise ValueError("size or duration of hop must be specified")
            self.overlap = self.window_length - self.hop_length
