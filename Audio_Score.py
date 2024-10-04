import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
from matplotlib.colors import LinearSegmentedColormap


def get_audio_file_paths(audio_directory: str) -> list[str]:
    """
    Génère les chemins complets de tous les fichiers audio dans un répertoire.

    Parameters:
    audio_directory (str): Chemin vers le répertoire contenant les fichiers audio.

    Returns:
    list[str]: Liste des chemins complets des fichiers audio (mp3, wav, flac).
    """
    # Parcourir les fichiers audio dans le répertoire et créer la liste des chemins complets
    audio_file_paths = [
        os.path.join(audio_directory, filename)
        for filename in os.listdir(audio_directory)
        if filename.endswith('.mp3') or filename.endswith('.wav') or filename.endswith('.flac')
    ]
    return audio_file_paths


def calculate_audio_spectrogram(file_path: str) -> tuple[np.ndarray, int]:
    """
    Charge un fichier audio et calcule son spectrogramme en dB.

    Parameters:
    file_path (str): Chemin vers le fichier audio.

    Returns:
    tuple: Spectrogramme en dB (numpy.ndarray), fréquence d'échantillonnage (int)
    """
    # Charger le fichier audio avec sa fréquence d'échantillonnage native
    audio_signal, sample_rate = librosa.load(file_path, sr=None)
    
    # Calculer la transformée de Fourier à court terme (STFT)
    spectrogram = librosa.stft(audio_signal)
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
    
    # Retourner les valeurs du spectrogramme et la fréquence d'échantillonnage
    return spectrogram_db, sample_rate


def definir_threshold_par_echelle(spectrogram_db: np.ndarray, fraction: float = 0.1) -> float:
    """
    Définit un seuil en dB en fonction de l'échelle des valeurs du spectrogramme.

    Parameters:
    spectrogram_db (numpy.ndarray): Spectrogramme en dB.
    fraction (float): Fraction de la plage d'amplitude pour définir le seuil (entre 0 et 1).

    Returns:
    float: Seuil en dB basé sur une fraction de l'échelle des amplitudes.
    """
    # Trouver les valeurs minimale et maximale du spectrogramme
    max_value = np.max(spectrogram_db)
    min_value = np.min(spectrogram_db)
    
    # Calculer le seuil basé sur une fraction de l'échelle, au-dessus de la valeur minimale
    threshold_db = min_value + fraction * (max_value - min_value)
    
    return threshold_db


def quantifier_distribution_points_par_freq(spectrogram_db: np.ndarray, sample_rate: int, threshold_db: float, percentiles: list[int] = [25, 50, 75, 90]) -> dict[int, float]:
    """
    Quantifie la distribution des points présents dans le spectrogramme par rapport aux fréquences.

    Parameters:
    spectrogram_db (numpy.ndarray): Spectrogramme en dB.
    sample_rate (int): Fréquence d'échantillonnage.
    threshold_db (float): Seuil dB pour considérer qu'une valeur est significative (utilisé pour filtrer le bruit).
    percentiles (list): Liste des percentiles à calculer (entre 0 et 100).

    Returns:
    dict: Dictionnaire où chaque percentile est associé à une fréquence (en Hz).
    """
    # Obtenir le nombre total de bins de fréquence et de temps
    num_freq_bins, num_time_frames = spectrogram_db.shape

    # Calculer le nombre de fenêtres temporelles où chaque fréquence a une valeur significative
    significant_counts = np.sum(spectrogram_db > threshold_db, axis=1)

    # Calculer les fréquences associées aux bins
    freqs = np.linspace(0, sample_rate / 2, num_freq_bins)

    # Normaliser les comptages pour avoir une distribution cumulative entre 0 et 1
    cumulative_counts = np.cumsum(significant_counts)
    cumulative_counts_normalized = cumulative_counts / cumulative_counts[-1]

    # Calculer les fréquences correspondant aux percentiles
    frequency_percentiles = {}

    for p in percentiles:
        # Trouver l'indice où le cumul des comptages atteint ou dépasse le percentile donné
        index = np.argmax(cumulative_counts_normalized >= p / 100.0)
        frequency_percentiles[p] = freqs[index]

    return frequency_percentiles

def transformer_quantiles_en_dataframe(titles_quantiles_dict: dict[str, dict[int, float]]) -> pd.DataFrame:
    """
    Transforme un dictionnaire de quantiles de fréquences pour chaque fichier audio en un DataFrame.

    Parameters:
    titles_quantiles_dict (dict): Dictionnaire où chaque clé est un titre de fichier et chaque valeur est un 
                                  autre dictionnaire des quantiles associés (ex: {25: freq, 50: freq, ...}).

    Returns:
    pandas.DataFrame: DataFrame avec chaque titre en ligne et les quantiles comme colonnes, arrondis à l'entier le plus proche.
    """
    # Créer une liste des titres (fichiers audio)
    titles = list(titles_quantiles_dict.keys())
    
    # Créer une liste des dictionnaires de quantiles (chaque entrée est pour un titre)
    quantiles_data = [titles_quantiles_dict[title] for title in titles]
    
    # Transformer en DataFrame
    df = pd.DataFrame(quantiles_data, index=titles)
    
    # Renommer l'index pour qu'il indique le nom des fichiers
    df.index.name = "Titre du fichier"
    
    # Convertir les valeurs en entiers
    df = df.astype(int)

    return df


def analyser_points_rupture(df_quantiles: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse le DataFrame des quantiles pour identifier les points de rupture en utilisant la première et la deuxième dérivée.
    Retourne un DataFrame contenant six colonnes : "Threshold Dérivée 1", "Second Threshold Dérivée 1", "Third Threshold Dérivée 1",
    "Threshold Dérivée 2", "Second Threshold Dérivée 2", et "Third Threshold Dérivée 2".

    Parameters:
    df_quantiles (pandas.DataFrame): DataFrame contenant les quantiles de fréquences pour chaque fichier audio.

    Returns:
    pandas.DataFrame: DataFrame avec six colonnes indiquant les quantiles sélectionnés par les points de rupture des dérivées 1 et 2, ainsi que les deuxièmes et troisièmes points les plus hauts.
    """
    # Calculer la première différence (dérivée 1)
    first_diff = df_quantiles.diff(axis=1)

    # Calculer la deuxième différence (dérivée 2)
    second_diff = first_diff.diff(axis=1)

    # Créer un DataFrame vide pour stocker les résultats
    df_thresholds = pd.DataFrame(index=df_quantiles.index)

    # Calculer le seuil pour la première dérivée (point maximum)
    max_first_diff_idx = first_diff.idxmax(axis=1)
    df_thresholds['Threshold Dérivée 1'] = max_first_diff_idx.apply(lambda x: df_quantiles.columns[df_quantiles.columns.get_loc(x) - 1] if df_quantiles.columns.get_loc(x) > 0 else df_quantiles.columns[0])

    # Calculer le deuxième point le plus haut pour la première dérivée
    second_max_first_diff_idx = first_diff.apply(lambda row: row.nlargest(2).idxmin(), axis=1)
    df_thresholds['Second Threshold Dérivée 1'] = second_max_first_diff_idx.apply(lambda x: df_quantiles.columns[df_quantiles.columns.get_loc(x) - 1] if df_quantiles.columns.get_loc(x) > 0 else df_quantiles.columns[0])

    # Calculer le troisième point le plus haut pour la première dérivée
    third_max_first_diff_idx = first_diff.apply(lambda row: row.nlargest(3).idxmin(), axis=1)
    df_thresholds['Third Threshold Dérivée 1'] = third_max_first_diff_idx.apply(lambda x: df_quantiles.columns[df_quantiles.columns.get_loc(x) - 1] if df_quantiles.columns.get_loc(x) > 0 else df_quantiles.columns[0])

    # Calculer le seuil pour la deuxième dérivée (point maximum)
    max_second_diff_idx = second_diff.idxmax(axis=1)
    df_thresholds['Threshold Dérivée 2'] = max_second_diff_idx.apply(lambda x: df_quantiles.columns[df_quantiles.columns.get_loc(x) - 1] if df_quantiles.columns.get_loc(x) > 0 else df_quantiles.columns[0])

    # Calculer le deuxième point le plus haut pour la deuxième dérivée
    second_max_second_diff_idx = second_diff.apply(lambda row: row.nlargest(2).idxmin(), axis=1)
    df_thresholds['Second Threshold Dérivée 2'] = second_max_second_diff_idx.apply(lambda x: df_quantiles.columns[df_quantiles.columns.get_loc(x) - 1] if df_quantiles.columns.get_loc(x) > 0 else df_quantiles.columns[0])

    # Calculer le troisième point le plus haut pour la deuxième dérivée
    third_max_second_diff_idx = second_diff.apply(lambda row: row.nlargest(3).idxmin(), axis=1)
    df_thresholds['Third Threshold Dérivée 2'] = third_max_second_diff_idx.apply(lambda x: df_quantiles.columns[df_quantiles.columns.get_loc(x) - 1] if df_quantiles.columns.get_loc(x) > 0 else df_quantiles.columns[0])

    return df_thresholds


def extract_average_significant_quantiles(df_quantiles: pd.DataFrame, df_rupture_threshold: pd.DataFrame) -> pd.Series:
    """
    Extrait la moyenne des quantiles significatifs de df_quantiles basés sur les seuils identifiés dans df_rupture_threshold.

    Parameters:
    df_quantiles (pandas.DataFrame): DataFrame contenant les quantiles de fréquences pour chaque fichier audio.
    df_rupture_threshold (pandas.DataFrame): DataFrame contenant les quantiles sélectionnés par les points de rupture.

    Returns:
    pandas.Series: Série contenant la moyenne des quantiles significatifs pour chaque fichier audio.
    """
    # Initialiser un dictionnaire pour stocker la moyenne des quantiles significatifs
    average_quantiles_dict = {}

    for index, row in df_rupture_threshold.iterrows():
        # Extraire les colonnes correspondant aux quantiles identifiés
        quantiles_list = []
        
        for col in row:
            if col in df_quantiles.columns:
                quantiles_list.append(df_quantiles.at[index, col])
        
        # Calculer la moyenne des quantiles extraits
        if quantiles_list:
            average_quantiles_dict[index] = np.mean(quantiles_list)
        else:
            average_quantiles_dict[index] = np.nan  # Si aucune valeur n'a été extraite

    # Transformer le dictionnaire en une série
    average_quantiles_series = pd.Series(average_quantiles_dict, name="Moyenne des quantiles significatifs")

    # Renommer l'index pour qu'il indique le nom des fichiers
    average_quantiles_series.index.name = "Titre du fichier"

    return average_quantiles_series


def calculate_degradation_score(average_quantiles_series: pd.Series, reference_value: float = 22000, title_order: bool = False) -> pd.Series:
    """
    Calcule le score de dégradation en comparant chaque moyenne des quantiles avec une valeur de référence.

    Parameters:
    average_quantiles_series (pandas.Series): Série contenant la moyenne des quantiles significatifs pour chaque fichier audio.
    reference_value (float): Valeur de référence à utiliser pour calculer la différence en pourcentage.
    title_order (bool): Si True, trie par ordre alphabétique des titres. Sinon, trie par valeur décroissante.

    Returns:
    pandas.Series: Série contenant le score de dégradation pour chaque fichier audio, triée selon le paramètre title_order.
    """
    # Calculer la différence en pourcentage par rapport à la valeur de référence
    percentage_diff = ((average_quantiles_series - reference_value) / reference_value) * 100

    # Appliquer les conditions pour le score de dégradation
    degradation_score = percentage_diff.apply(lambda x: 0 if x >= 0 else abs(x))

    degradation_score_rounded = degradation_score.round(2)

    # Renommer la série pour indiquer qu'elle contient le score de dégradation
    degradation_score_rounded.name = "Degradation Score"

    # Trier la série selon le paramètre title_order
    if title_order:
        degradation_score_sorted = degradation_score_rounded.sort_index(ascending=True)
    else:
        degradation_score_sorted = degradation_score_rounded.sort_values(ascending=False)

    return degradation_score_sorted


def analyser_fichiers_audio(audio_file_paths: str, fraction: float = 0.1, percentiles: list[int] = [80, 81, 82, 83, 84, 85, 86, 87, 88 ,89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]) -> pd.DataFrame:
    """
    Parcourt les fichiers audio dans un répertoire, calcule les spectrogrammes et les quantiles de fréquences, 
    et retourne un DataFrame avec chaque fichier audio et les quantiles associés.

    Parameters:
    audio_file_paths (str): Chemin vers le répertoire contenant les fichiers audio.
    fraction (float): Fraction de la plage d'amplitude pour définir le seuil (entre 0 et 1).
    percentiles (list): Liste des percentiles à calculer (entre 0 et 100).

    Returns:
    pandas.DataFrame: DataFrame contenant les titres des fichiers audio et les quantiles associés.
    """

    # Dictionnaire pour stocker les quantiles de chaque fichier
    titles_quantiles_dict = {}

    # Parcourir les fichiers audio
    for file_path in audio_file_paths:

        filename = os.path.basename(file_path)

        # Calculer les valeurs du spectrogramme
        spectrogram_db, sample_rate = calculate_audio_spectrogram(file_path)

        # Définir le seuil automatiquement en fonction de l'échelle
        threshold_db_value = definir_threshold_par_echelle(spectrogram_db, fraction=fraction)

        # Quantifier les fréquences correspondant aux percentiles de distribution des points
        frequency_percentiles = quantifier_distribution_points_par_freq(spectrogram_db, sample_rate, threshold_db=threshold_db_value, percentiles=percentiles)

        # Ajouter les quantiles au dictionnaire, avec le nom du fichier comme clé
        titles_quantiles_dict[filename] = frequency_percentiles

    # Transformer le dictionnaire en DataFrame
    df_quantiles = transformer_quantiles_en_dataframe(titles_quantiles_dict)

    df_rupture_treshold = analyser_points_rupture(df_quantiles)

    # Utiliser la nouvelle fonction pour extraire les quantiles significatifs
    average_quantiles_per_title = extract_average_significant_quantiles(df_quantiles, df_rupture_treshold)

    # Utiliser la nouvelle fonction pour calculer le score de dégradation pour chaque titre
    degradation_score_per_title = calculate_degradation_score(average_quantiles_per_title)

    return degradation_score_per_title


def plot_spectrogram(spectrogram_db: np.ndarray, sample_rate: int, filename: str) -> None:
    """
    Affiche un spectrogramme d'un fichier audio.

    Parameters:
    spectrogram_db (numpy.ndarray): Spectrogramme en dB.
    sample_rate (int): Fréquence d'échantillonnage.
    audio_directory (str): Répertoire contenant le fichier audio.
    filename (str): Nom du fichier audio.
    """

    # Visualiser le spectrogramme
    plt.figure(figsize=(10, 6), facecolor='black')  # Fenêtre avec fond noir
    ax = plt.gca()
    
    librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis='time', y_axis='hz', cmap=custom_cmap)
    cbar = plt.colorbar(format="%+2.0f dB", cmap=custom_cmap)
    cbar.ax.yaxis.set_tick_params(color='white')  # Changer la couleur des graduations de la barre de couleurs
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')  # Changer la couleur des étiquettes de la barre de couleurs

    # Configuration des titres et étiquettes
    plt.title(f"{filename}", color='white')
    plt.xlabel("Temps (s)", color='white')
    plt.ylabel("Fréquence (Hz)", color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')

    # Ajouter un contour blanc autour du graphique
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Ajouter un contour blanc autour de la barre de couleurs
    cbar.outline.set_edgecolor('white')

    plt.tight_layout()
    plt.show()


def analyse_and_plot(audio_directory: str, filename: str, filename_2: str,
                     calculate_degradation: bool = False, plot_spectrogram_1: bool = False, plot_spectrogram_2: bool = False) -> None:
    """
    Fusionne les fonctions pour calculer le score de dégradation et afficher les spectrogrammes.

    Parameters:
    audio_directory (str): Répertoire contenant les fichiers audio.
    filename (str): Nom du premier fichier audio.
    filename_2 (str): Nom du second fichier audio.
    calculate_degradation (bool): Si True, calcule et affiche le score de dégradation pour chaque fichier audio.
    plot_spectrogram_1 (bool): Si True, affiche le spectrogramme du premier fichier.
    plot_spectrogram_2 (bool): Si True, affiche le spectrogramme du second fichier.
    """
    # Créer les chemins complets des fichiers
    audio_file_paths = get_audio_file_paths(audio_directory)
    file_path_1 = os.path.join(audio_directory, filename)
    file_path_2 = os.path.join(audio_directory, filename_2)

    # Calculer et afficher le score de dégradation si demandé
    if calculate_degradation:
        degradation_score_per_title = analyser_fichiers_audio(audio_file_paths)
        print(degradation_score_per_title)

    # Afficher le spectrogramme du premier fichier si demandé
    if plot_spectrogram_1:
        spectrogram_db_1, sample_rate_1 = calculate_audio_spectrogram(file_path_1)
        plot_spectrogram(spectrogram_db_1, sample_rate_1, filename)

    # Afficher le spectrogramme du second fichier si demandé
    if plot_spectrogram_2:
        spectrogram_db_2, sample_rate_2 = calculate_audio_spectrogram(file_path_2)
        plot_spectrogram(spectrogram_db_2, sample_rate_2, filename_2)

    
# Colormap personnalisée
colors = ["#000000", "#0000FF", "#008000", "#FFFF00", "#FF0000"]  # Noir, Bleu, Vert, Jaune, Rouge
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# Répertoire contenant les fichiers audio
audio_directory = 'D:\\MusicDJ'
filename = 'Grupo La Cumbia - Cumbia Buena (Intro) 95.mp3'
filename_2 = 'Watussi Jowell Y Randy Nengo Flow - Dale Pal Piso Dj Matt Break Acapella Hype Outro 96.mp3'


# Exemple d'utilisation de la fonction fusionnée
analyse_and_plot(
    audio_directory=audio_directory,
    filename=filename,
    filename_2=filename_2,
    calculate_degradation=True,
    plot_spectrogram_1=False,
    plot_spectrogram_2=False
)