import audio_score

# Directory containing the audio files
audio_directory = "C:\\Users\\tibo\\Documents\\music"
filename = "Grupo La Cumbia - Cumbia Buena (Intro) 95.mp3"
filename_2 = "Watussi Jowell Y Randy Nengo Flow - Dale Pal Piso Dj Matt Break Acapella Hype Outro 96.mp3"


# Example of using the merged function
audio_score.analyse_and_plot(
    audio_directory=audio_directory,
    filename=filename,
    filename_2=filename_2,
    calculate_degradation=True,
    plot_spectrogram_1=False,
    plot_spectrogram_2=False,
)
