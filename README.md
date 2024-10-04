
# Audio File Analyzer

This Python project allows for the analysis and visualization of audio files. The script can calculate degradation scores based on frequency quantiles and generate spectrograms for selected audio files. It also supports multiple audio formats (`.mp3`, `.wav`, `.flac`).

The higher the score, the worse the quality.

## Features

- **Audio Spectrogram Generation**: Calculates and visualizes the spectrogram (in dB) for audio files.
- **Degradation Score Calculation**: Computes a degradation score for each audio file based on frequency quantile analysis and compares it to a reference value (default is 22,000).
- **Quantile Extraction**: Extracts frequency quantiles and performs breakpoint analysis to identify significant changes in frequency distribution.
- **Multiple Audio Formats Supported**: The analyzer supports `.mp3`, `.wav`, and `.flac` file formats.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/audio-file-analyzer.git
   ```

2. Navigate to the project directory:
   ```bash
   cd audio-file-analyzer
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

The following Python libraries are required:

- `numpy`
- `librosa`
- `matplotlib`
- `pandas`
- `os`

To install them, use:
```bash
pip install numpy librosa matplotlib pandas
```

## Usage

### 1. Analyze and Plot Audio Files

You can analyze audio files by calculating degradation scores or generating spectrograms. Use the `analyse_and_plot` function to perform both actions.

```python
analyse_and_plot(
    audio_directory='path_to_audio_directory',
    filename='filename_1.mp3',
    filename_2='filename_2.mp3',
    calculate_degradation=True,        # Set to True to compute degradation scores
    plot_spectrogram_1=True,           # Set to True to plot the first file's spectrogram
    plot_spectrogram_2=False           # Set to True to plot the second file's spectrogram
)
```

#### Example
```python
analyse_and_plot(
    audio_directory='D:\MusicDJ',
    filename='Grupo La Cumbia - Cumbia Buena (Intro) 95.mp3',
    filename_2='Watussi Jowell Y Randy Nengo Flow - Dale Pal Piso.mp3',
    calculate_degradation=True,
    plot_spectrogram_1=True,
    plot_spectrogram_2=True
)
```

### 2. Functions Overview

- `get_audio_file_paths(audio_directory: str) -> list[str]`: Retrieves all audio file paths from the specified directory.
- `calculate_audio_spectrogram(file_path: str) -> tuple[np.ndarray, int]`: Calculates the spectrogram of an audio file.
- `set_threshold_by_scale(spectrogram_db: np.ndarray, fraction: float = 0.1) -> float`: Sets the threshold for spectrogram values based on the amplitude scale.
- `quantify_distribution_points_by_frequency(spectrogram_db: np.ndarray, sample_rate: int, threshold_db: float, percentiles: list[int]) -> dict[int, float]`: Quantifies the frequency distribution points in a spectrogram.
- `transform_quantiles_to_dataframe(titles_quantiles_dict: dict[str, dict[int, float]]) -> pd.DataFrame`: Converts a dictionary of frequency quantiles into a DataFrame.
- `analyze_breakpoints(df_quantiles: pd.DataFrame) -> pd.DataFrame`: Analyzes frequency breakpoints using first and second derivatives.
- `extract_average_significant_quantiles(df_quantiles: pd.DataFrame, df_breakpoint_threshold: pd.DataFrame) -> pd.Series`: Extracts the average significant quantiles from a DataFrame.
- `calculate_degradation_score(average_quantiles_series: pd.Series, reference_value: float = 22000, title_order: bool = False) -> pd.Series`: Calculates a degradation score based on quantiles and a reference value.
- `analyze_audio_files(audio_file_paths: list[str], fraction: float, percentiles: list[int]) -> pd.DataFrame`: Analyzes the audio files in the specified directory and returns a DataFrame with their degradation scores.

### 3. Custom Colormap

The script uses a custom colormap for spectrogram visualization:
```python
colors = ["#000000", "#0000FF", "#008000", "#FFFF00", "#FF0000"]  # Black, Blue, Green, Yellow, Red
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Author

- [Stettler Thibaud](https://github.com/your-username)
