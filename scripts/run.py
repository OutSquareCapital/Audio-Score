from pathlib import Path

import audio_score

# Spécifiez directement le chemin du fichier audio ici
AUDIO_FILE = (
    Path()
    .resolve()
    .parent.joinpath(
        "db",
        "music",
        "Eddy Herrera - Tu Eres Ajena (Dj Jesus Olivera Intro Outro) 135bpm 135",
    )
    .with_suffix(".mp3")
)
print(AUDIO_FILE)
# Analyse simple
result = audio_score.analyze_audio_file(AUDIO_FILE)

# Affiche les résultats
print(f"Analyse de {AUDIO_FILE.name}")
print(f"Score de qualité: {result.quality_score:.2f}% (0% = qualité parfaite)")
print(f"Taux d'échantillonnage: {result.sample_rate} Hz")
print(f"Spectrogramme (dB): {result.spectrogram_db.shape}")
