import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt
from filtrado import ButterBandpassFilter


# ------------------------------
# Parámetros generales
# ------------------------------
FS = 250            # Frecuencia de muestreo (Hz)
FASE_SEC = 2        # Duración de cada ventana (s)
SAMPLES_PER_FASE = FS * FASE_SEC
INPUT_FOLDER = "./ensayos"
OUTPUT_FILE = "./dataset_final_canal2.csv"


# ------------------------------
# Configurar filtros
# ------------------------------
filtro = ButterBandpassFilter(20, 80, fs=FS, order=4)


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


# ------------------------------
# Preprocesamiento EMG
# ------------------------------
def preprocess_emg(signal):
    # Filtro pasa banda
    filtered = filtro.apply(signal)
    # Rectificación cuadrática
    rectified = filtered ** 2
    # Normalización entre 0 y 1
    normalized = (rectified - np.min(rectified)) / (np.max(rectified) - np.min(rectified) + 1e-8)
    # Suavizado con pasa bajo (4 Hz)
    smoothed = butter_lowpass_filter(normalized, cutoff=4, fs=FS)
    return smoothed


# ------------------------------
# Procesamiento de todos los archivos
# ------------------------------
all_windows = []
factor = (4.5 / ((2**23 - 1) * 24)) * 16  # factor de conversión a microvoltios


for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".csv"):
        filepath = os.path.join(INPUT_FOLDER, filename)
        df = pd.read_csv(filepath)


        # Convertir canal 2 a microvoltios
        df["ch2_uV"] = df["ch2"] * factor


        # Aplicar preprocesamiento EMG
        df["ch2_proc"] = preprocess_emg(df["ch2_uV"])


        # Crear vector de tiempo (opcional)
        time = np.linspace(0, len(df) / FS, len(df))
        df["time"] = time


        # Segmentar en ventanas de 2 segundos y etiquetar
        num_fases = len(df) // SAMPLES_PER_FASE
        for j in range(num_fases):
            start = j * SAMPLES_PER_FASE
            end = start + SAMPLES_PER_FASE
            ventana = df.iloc[start:end][["time", "ch2_proc"]].copy()
            ventana["etiqueta"] = "contraccion" if j % 2 == 0 else "relajacion"
            ventana["archivo_origen"] = filename
            all_windows.append(ventana)


# ------------------------------
# Guardar dataset final
# ------------------------------
dataset_final = pd.concat(all_windows, ignore_index=True)
dataset_final.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Dataset final guardado en {OUTPUT_FILE} con {len(dataset_final)} filas.")
