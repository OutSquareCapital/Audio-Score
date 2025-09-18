from pathlib import Path
from typing import NamedTuple

import librosa
import numpy as np
from numpy.typing import NDArray


class AudioAnalysisResult(NamedTuple):
    spectrogram_db: NDArray[np.floating]
    sample_rate: int
    quality_score: float


def _threshold(spectrogram_db: NDArray[np.floating]) -> float:
    max_val = float(np.max(spectrogram_db))
    min_val = float(np.min(spectrogram_db))
    return min_val + 0.1 * (max_val - min_val)


def analyze_audio_file(file_path: Path) -> AudioAnalysisResult:
    """
    Analyze a single audio file and return its spectrogram and quality score

    Args:
        file_path: Path to an audio file

    Returns:
        Tuple containing:
        - spectrogram_db: The audio spectrogram in dB
        - sample_rate: The audio sample rate
        - quality_score: Score from 0-100 indicating audio quality (0 is best)
    """
    # Load the audio file with its native sample rate
    audio_signal, sample_rate = librosa.load(file_path, sr=None)  # type: ignore
    sample_rate = int(sample_rate)  # Convert to int if it's a float

    # Compute the spectrogram
    spectrogram: NDArray[np.floating] = librosa.stft(audio_signal)  # type: ignore
    spectrogram_db: NDArray[np.floating] = librosa.amplitude_to_db(abs(spectrogram))  # type: ignore
    print("librosa analysis done")
    # Calculate simple quality score based on frequency distribution
    # Get the number of frequency bins
    num_freq_bins: float = spectrogram_db.shape[0]

    # Calculate frequency distribution
    significant_counts = np.sum(spectrogram_db > _threshold(spectrogram_db), axis=1)
    freqs = np.linspace(0, sample_rate / 2, num_freq_bins)
    highest_idx = np.max(np.where(significant_counts > 0)[0])
    highest_freq = freqs[highest_idx]

    # Get highest frequency with significant content
    if significant_counts.any():
        # Calculate quality score (0-100) with 0 being perfect quality
        # A perfect file would have frequencies up to 20-22kHz
        reference = 22000
        score = 0.0
        if freqs[np.max(np.where(significant_counts > 0)[0])] < reference:
            score = ((reference - highest_freq) / reference) * 100
    else:
        score = 100.0  # No significant frequencies found

    return AudioAnalysisResult(spectrogram_db, sample_rate, score)
