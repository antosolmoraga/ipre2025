import pandas as pd
import matplotlib.pyplot as plt
from filtrado import ButterBandpassFilter
import numpy as np
from scipy.signal import butter, filtfilt

FS=250
filtro = ButterBandpassFilter(20, 80, fs=FS, order=4)
factor=(4.5/((2**23-1)*24))*16

# Cargar datos
df = pd.read_csv("subject01_session04_data.csv")


for i in range(1,9):
    df[f"ch{i}_uV"]=df[f"ch{i}"]*factor

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


# --- Preprocesamiento y graficado ---
time = np.linspace(0, len(df) / FS, len(df))
plt.figure(figsize=(10, 4))


# Filtrado pasa banda
filtered = filtro.apply(df[f"ch{2}_uV"])

# Rectificación
rectified = filtered**2

# Normalización entre 0 y 1
normalized = (rectified - np.min(rectified)) / (np.max(rectified) - np.min(rectified))

smoothed= butter_lowpass_filter(normalized, cutoff=10, fs=FS)

# Graficar
plt.plot(time, smoothed + 2*2, label=f'Canal {2}')  
# ↑ sumo i*2 para desplazar verticalmente cada canal y que no se superpongan


plt.legend()
plt.title("Señales EMG canal 2")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (µV)")
plt.savefig("emg_canal2_procesadas.png")
