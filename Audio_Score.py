import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
from matplotlib.colors import LinearSegmentedColormap


def get_audio_file_paths(audio_directory: str) -> list[str]:
    """
    Generates the full paths of all audio files in a directory.

    Parameters:
    audio_directory (str): Path to the directory containing audio files.

    Returns:
    list[str]: List of full paths to audio files (mp3, wav, flac).
    """
    # Iterate through the audio files in the directory and create a list of full paths
    audio_file_paths = [
        os.path.join(audio_directory, filename)
        for filename in os.listdir(audio_directory)
        if filename.endswith('.mp3') or filename.endswith('.wav') or filename.endswith('.flac')
    ]
    return audio_file_paths

def calculate_audio_spectrogram(file_path: str) -> tuple[np.ndarray, int]:
    """
    Loads an audio file and calculates its spectrogram in dB.

    Parameters:
    file_path (str): Path to the audio file.

    Returns:
    tuple: Spectrogram in dB (numpy.ndarray), sample rate (int)
    """
    # Load the audio file with its native sample rate
    audio_signal, sample_rate = librosa.load(file_path, sr=None)
    
    # Compute the Short-Time Fourier Transform (STFT)
    spectrogram = librosa.stft(audio_signal)
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
    
    # Return the spectrogram and the sample rate
    return spectrogram_db, sample_rate


def definir_threshold_par_echelle(spectrogram_db: np.ndarray, fraction: float = 0.1) -> float:
    """
    Defines a dB threshold based on the scale of spectrogram values.

    Parameters:
    spectrogram_db (numpy.ndarray): Spectrogram in dB.
    fraction (float): Fraction of the amplitude range to define the threshold (between 0 and 1).

    Returns:
    float: Threshold in dB based on a fraction of the amplitude scale.
    """
    # Find the minimum and maximum values in the spectrogram
    max_value = np.max(spectrogram_db)
    min_value = np.min(spectrogram_db)
    
    # Calculate the threshold based on a fraction of the range, above the minimum value
    threshold_db = min_value + fraction * (max_value - min_value)
    
    return threshold_db


def quantifier_distribution_points_par_freq(spectrogram_db: np.ndarray, sample_rate: int, threshold_db: float, percentiles: list[int] = [25, 50, 75, 90]) -> dict[int, float]:
    """
    Quantifies the distribution of significant points in the spectrogram relative to frequencies.

    Parameters:
    spectrogram_db (numpy.ndarray): Spectrogram in dB.
    sample_rate (int): Sampling rate.
    threshold_db (float): dB threshold for considering a value significant (used to filter noise).
    percentiles (list): List of percentiles to calculate (between 0 and 100).

    Returns:
    dict: Dictionary where each percentile is associated with a frequency (in Hz).
    """
    # Get the total number of frequency bins and time frames
    num_freq_bins, num_time_frames = spectrogram_db.shape

    # Calculate the number of time windows where each frequency has a significant value
    significant_counts = np.sum(spectrogram_db > threshold_db, axis=1)

    # Calculate the frequencies associated with the bins
    freqs = np.linspace(0, sample_rate / 2, num_freq_bins)

    # Normalize the counts to create a cumulative distribution between 0 and 1
    cumulative_counts = np.cumsum(significant_counts)
    cumulative_counts_normalized = cumulative_counts / cumulative_counts[-1]

    # Calculate the frequencies corresponding to the percentiles
    frequency_percentiles = {}

    for p in percentiles:
        # Find the index where the cumulative counts reach or exceed the given percentile
        index = np.argmax(cumulative_counts_normalized >= p / 100.0)
        frequency_percentiles[p] = freqs[index]

    return frequency_percentiles

def transformer_quantiles_en_dataframe(titles_quantiles_dict: dict[str, dict[int, float]]) -> pd.DataFrame:
    """
    Transforms a dictionary of frequency quantiles for each audio file into a DataFrame.

    Parameters:
    titles_quantiles_dict (dict): Dictionary where each key is a file title, and each value is another dictionary of associated quantiles (e.g., {25: freq, 50: freq, ...}).

    Returns:
    pandas.DataFrame: DataFrame with each title as a row and quantiles as columns, rounded to the nearest integer.
    """
    # Create a list of file titles
    titles = list(titles_quantiles_dict.keys())
    
    # Create a list of quantile dictionaries (each entry is for a title)
    quantiles_data = [titles_quantiles_dict[title] for title in titles]
    
    # Convert to DataFrame
    df = pd.DataFrame(quantiles_data, index=titles)
    
    # Rename the index to indicate the file names
    df.index.name = "File Title"
    
    # Convert the values to integers
    df = df.astype(int)

    return df


def analyser_points_rupture(df_quantiles: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the quantiles DataFrame to identify breakpoints using the first and second derivatives.
    Returns a DataFrame with six columns: "First Derivative Threshold", "Second Threshold First Derivative", "Third Threshold First Derivative",
    "Second Derivative Threshold", "Second Threshold Second Derivative", and "Third Threshold Second Derivative".

    Parameters:
    df_quantiles (pandas.DataFrame): DataFrame containing the frequency quantiles for each audio file.

    Returns:
    pandas.DataFrame: DataFrame with six columns indicating the quantiles selected by the first and second derivative breakpoints, along with the second and third highest points.
    """
    # Calculate the first difference (first derivative)
    first_diff = df_quantiles.diff(axis=1)

    # Calculate the second difference (second derivative)
    second_diff = first_diff.diff(axis=1)

    # Create an empty DataFrame to store the results
    df_thresholds = pd.DataFrame(index=df_quantiles.index)

    # Calculate the threshold for the first derivative (maximum point)
    max_first_diff_idx = first_diff.idxmax(axis=1)
    df_thresholds['First Derivative Threshold'] = max_first_diff_idx.apply(lambda x: df_quantiles.columns[df_quantiles.columns.get_loc(x) - 1] if df_quantiles.columns.get_loc(x) > 0 else df_quantiles.columns[0])

    # Calculate the second-highest point for the first derivative
    second_max_first_diff_idx = first_diff.apply(lambda row: row.nlargest(2).idxmin(), axis=1)
    df_thresholds['Second Threshold First Derivative'] = second_max_first_diff_idx.apply(lambda x: df_quantiles.columns[df_quantiles.columns.get_loc(x) - 1] if df_quantiles.columns.get_loc(x) > 0 else df_quantiles.columns[0])

    # Calculate the third-highest point for the first derivative
    third_max_first_diff_idx = first_diff.apply(lambda row: row.nlargest(3).idxmin(), axis=1)
    df_thresholds['Third Threshold First Derivative'] = third_max_first_diff_idx.apply(lambda x: df_quantiles.columns[df_quantiles.columns.get_loc(x) - 1] if df_quantiles.columns.get_loc(x) > 0 else df_quantiles.columns[0])

    # Calculate the threshold for the second derivative (maximum point)
    max_second_diff_idx = second_diff.idxmax(axis=1)
    df_thresholds['Second Derivative Threshold'] = max_second_diff_idx.apply(lambda x: df_quantiles.columns[df_quantiles.columns.get_loc(x) - 1] if df_quantiles.columns.get_loc(x) > 0 else df_quantiles.columns[0])

    # Calculate the second-highest point for the second derivative
    second_max_second_diff_idx = second_diff.apply(lambda row: row.nlargest(2).idxmin(), axis=1)
    df_thresholds['Second Threshold Second Derivative'] = second_max_second_diff_idx.apply(lambda x: df_quantiles.columns[df_quantiles.columns.get_loc(x) - 1] if df_quantiles.columns.get_loc(x) > 0 else df_quantiles.columns[0])

    # Calculate the third-highest point for the second derivative
    third_max_second_diff_idx = second_diff.apply(lambda row: row.nlargest(3).idxmin(), axis=1)
    df_thresholds['Third Threshold Second Derivative'] = third_max_second_diff_idx.apply(lambda x: df_quantiles.columns[df_quantiles.columns.get_loc(x) - 1] if df_quantiles.columns.get_loc(x) > 0 else df_quantiles.columns[0])

    return df_thresholds


def extract_average_significant_quantiles(df_quantiles: pd.DataFrame, df_rupture_threshold: pd.DataFrame) -> pd.Series:
    """
    Extracts the average significant quantiles from df_quantiles based on the thresholds identified in df_rupture_threshold.

    Parameters:
    df_quantiles (pandas.DataFrame): DataFrame containing the frequency quantiles for each audio file.
    df_rupture_threshold (pandas.DataFrame): DataFrame containing the quantiles selected by the breakpoint analysis.

    Returns:
    pandas.Series: Series containing the average significant quantiles for each audio file.
    """
    # Initialize a dictionary to store the average significant quantiles
    average_quantiles_dict = {}

    for index, row in df_rupture_threshold.iterrows():
        # Extract the columns corresponding to the identified quantiles
        quantiles_list = []
        
        for col in row:
            if col in df_quantiles.columns:
                quantiles_list.append(df_quantiles.at[index, col])
        
        # Calculate the average of the extracted quantiles
        if quantiles_list:
            average_quantiles_dict[index] = np.mean(quantiles_list)
        else:
            average_quantiles_dict[index] = np.nan  # If no values were extracted

    # Convert the dictionary to a Series
    average_quantiles_series = pd.Series(average_quantiles_dict, name="Average Significant Quantiles")

    # Rename the index to indicate the file names
    average_quantiles_series.index.name = "File Title"

    return average_quantiles_series


def calculate_degradation_score(average_quantiles_series: pd.Series, reference_value: float = 22000, title_order: bool = False) -> pd.Series:
    """
    Calculates the degradation score by comparing each average quantile with a reference value.

    Parameters:
    average_quantiles_series (pandas.Series): Series containing the average significant quantiles for each audio file.
    reference_value (float): Reference value to calculate the percentage difference.
    title_order (bool): If True, sorts by file title alphabetically. Otherwise, sorts by decreasing value.

    Returns:
    pandas.Series: Series containing the degradation score for each audio file, sorted according to the title_order parameter.
    """
    # Calculate the percentage difference from the reference value
    percentage_diff = ((average_quantiles_series - reference_value) / reference_value) * 100

    # Apply the conditions for the degradation score
    degradation_score = percentage_diff.apply(lambda x: 0 if x >= 0 else abs(x))

    degradation_score_rounded = degradation_score.round(2)

    # Rename the series to indicate it contains the degradation score
    degradation_score_rounded.name = "Degradation Score"

    # Sort the series based on the title_order parameter
    if title_order:
        degradation_score_sorted = degradation_score_rounded.sort_index(ascending=True)
    else:
        degradation_score_sorted = degradation_score_rounded.sort_values(ascending=False)

    return degradation_score_sorted


def analyser_fichiers_audio(audio_file_paths: str, fraction: float = 0.1, percentiles: list[int] = [80, 81, 82, 83, 84, 85, 86, 87, 88 ,89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]) -> pd.DataFrame:
    """
    Iterates through the audio files in a directory, calculates the spectrograms and frequency quantiles, 
    and returns a DataFrame with each audio file and its associated quantiles.

    Parameters:
    audio_file_paths (str): Paths to the directory containing audio files.
    fraction (float): Fraction of the amplitude range to define the threshold (between 0 and 1).
    percentiles (list): List of percentiles to calculate (between 0 and 100).

    Returns:
    pandas.DataFrame: DataFrame containing the file titles and the associated quantiles.
    """

    # Dictionary to store quantiles for each file
    titles_quantiles_dict = {}

    # Iterate through audio files
    for file_path in audio_file_paths:

        filename = os.path.basename(file_path)

        # Calculate the spectrogram values
        spectrogram_db, sample_rate = calculate_audio_spectrogram(file_path)

        # Define the threshold automatically based on the scale
        threshold_db_value = definir_threshold_par_echelle(spectrogram_db, fraction=fraction)

        # Quantify the frequencies corresponding to the percentiles of the point distribution
        frequency_percentiles = quantifier_distribution_points_par_freq(spectrogram_db, sample_rate, threshold_db=threshold_db_value, percentiles=percentiles)

        # Add the quantiles to the dictionary, using the file name as the key
        titles_quantiles_dict[filename] = frequency_percentiles

    # Convert the dictionary into a DataFrame
    df_quantiles = transformer_quantiles_en_dataframe(titles_quantiles_dict)

    df_rupture_treshold = analyser_points_rupture(df_quantiles)

    # Use the new function to extract the significant quantiles
    average_quantiles_per_title = extract_average_significant_quantiles(df_quantiles, df_rupture_treshold)

    # Use the new function to calculate the degradation score for each title
    degradation_score_per_title = calculate_degradation_score(average_quantiles_per_title)

    return degradation_score_per_title


def plot_spectrogram(spectrogram_db: np.ndarray, sample_rate: int, filename: str) -> None:
    """
    Displays a spectrogram of an audio file.

    Parameters:
    spectrogram_db (numpy.ndarray): Spectrogram in dB.
    sample_rate (int): Sampling rate.
    filename (str): Name of the audio file.
    """

    # Visualize the spectrogram
    plt.figure(figsize=(10, 6), facecolor='black')  # Window with a black background
    ax = plt.gca()
    
    librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis='time', y_axis='hz', cmap=custom_cmap)
    cbar = plt.colorbar(format="%+2.0f dB", cmap=custom_cmap)
    cbar.ax.yaxis.set_tick_params(color='white')  # Change the color of the colorbar ticks
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')  # Change the color of the colorbar labels

    # Configure titles and labels
    plt.title(f"{filename}", color='white')
    plt.xlabel("Time (s)", color='white')
    plt.ylabel("Frequency (Hz)", color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')

    # Add a white outline around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Add a white outline around the colorbar
    cbar.outline.set_edgecolor('white')

    plt.tight_layout()
    plt.show()


def analyse_and_plot(audio_directory: str, filename: str, filename_2: str,
                     calculate_degradation: bool = False, plot_spectrogram_1: bool = False, plot_spectrogram_2: bool = False) -> None:
    """
    Merges functions to calculate the degradation score and display spectrograms.

    Parameters:
    audio_directory (str): Directory containing the audio files.
    filename (str): Name of the first audio file.
    filename_2 (str): Name of the second audio file.
    calculate_degradation (bool): If True, calculates and displays the degradation score for each audio file.
    plot_spectrogram_1 (bool): If True, displays the spectrogram for the first file.
    plot_spectrogram_2 (bool): If True, displays the spectrogram for the second file.
    """
    # Create the full paths of the files
    audio_file_paths = get_audio_file_paths(audio_directory)
    file_path_1 = os.path.join(audio_directory, filename)
    file_path_2 = os.path.join(audio_directory, filename_2)

    # Calculate and display the degradation score if requested
    if calculate_degradation:
        degradation_score_per_title = analyser_fichiers_audio(audio_file_paths)
        print(degradation_score_per_title)

    # Display the spectrogram for the first file if requested
    if plot_spectrogram_1:
        spectrogram_db_1, sample_rate_1 = calculate_audio_spectrogram(file_path_1)
        plot_spectrogram(spectrogram_db_1, sample_rate_1, filename)

    # Display the spectrogram for the second file if requested
    if plot_spectrogram_2:
        spectrogram_db_2, sample_rate_2 = calculate_audio_spectrogram(file_path_2)
        plot_spectrogram(spectrogram_db_2, sample_rate_2, filename_2)

    
# Custom colormap
colors = ["#000000", "#0000FF", "#008000", "#FFFF00", "#FF0000"]  # Black, Blue, Green, Yellow, Red
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# Directory containing the audio files
audio_directory = 'D:\\MusicDJ'
filename = 'Grupo La Cumbia - Cumbia Buena (Intro) 95.mp3'
filename_2 = 'Watussi Jowell Y Randy Nengo Flow - Dale Pal Piso Dj Matt Break Acapella Hype Outro 96.mp3'


# Example of using the merged function
analyse_and_plot(
    audio_directory=audio_directory,
    filename=filename,
    filename_2=filename_2,
    calculate_degradation=True,
    plot_spectrogram_1=False,
    plot_spectrogram_2=False
)
