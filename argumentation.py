import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt
from filtrado import ButterHighpassFilter
import matplotlib.pyplot as plt

# ------------------------------
# Parámetros generales
# ------------------------------
FS = 250
FASE_SEC = 2
SAMPLES_PER_FASE = int(0.5 * FS)
INPUT_FOLDER = "./ensayos"
OUTPUT_FILE = "./dataset_final_canal2_baseline.csv"


# ------------------------------
# Configurar filtros
# ------------------------------
filtro = ButterHighpassFilter(20, fs=FS, order=4)


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


# ------------------------------
# Preprocesamiento EMG
# ------------------------------
def preprocess_emg(signal):
    filtered = filtro.apply(signal)
    rectified = filtered ** 2
    normalized = (rectified - np.min(rectified)) / (np.max(rectified) - np.min(rectified) + 1e-8)
    smoothed = butter_lowpass_filter(normalized, cutoff=4, fs=FS)
    return smoothed


# ------------------------------
# Procesamiento de todos los archivos
# ------------------------------
all_windows = []
factor = (4.5 / ((2**23 - 1) * 24)) * 16  # Conversión a microvoltios


for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".csv"):
        filepath = os.path.join(INPUT_FOLDER, filename)
        df = pd.read_csv(filepath)

        # Convertir canal 2 a microvoltios
        df["ch2_uV"] = df["ch2"] * factor

        # Aplicar preprocesamiento
        df["ch2_proc"] = preprocess_emg(df["ch2_uV"])

        # eliminar primer 0.5 segundo 
        df = df.iloc[125:].reset_index(drop=True)

        # Calcular baseline
        baseline_window = int(1 * FS)
        baseline_section = df["ch2_proc"].iloc[:baseline_window]
        baseline_value=baseline_section.mean()
        baseline_std=baseline_section.std()

        ###### restar baseline
        df["ch2_proc"] = df["ch2_proc"] - baseline_value

        # Crear vector de tiempo
        df["time"] = np.linspace(0, len(df) / FS, len(df))

        ##### calcular umbral
        threshold = 1.5 * baseline_std

        # Segmentar en ventanas
        num_fases = len(df) // SAMPLES_PER_FASE
        for j in range(num_fases):
            start = j * SAMPLES_PER_FASE
            end = start + SAMPLES_PER_FASE
            ventana = df.iloc[start:end][["time", "ch2_proc"]].copy()

            # Calcular media de la ventana
            mean_val = ventana["ch2_proc"].mean()
            #####threshold=baseline_value + 1.5*baseline_std

            # Etiquetar respecto al baseline
            etiqueta = "contraccion" if mean_val > threshold else "relajacion"
            
   
            ventana["etiqueta"] = etiqueta
            ventana["archivo_origen"] = filename
            all_windows.append(ventana)


# ------------------------------
# Guardar dataset final
# ------------------------------
dataset_final = pd.concat(all_windows, ignore_index=True)
dataset_final.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Dataset final guardado en {OUTPUT_FILE} con {len(dataset_final)} filas.")
