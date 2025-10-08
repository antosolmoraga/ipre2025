import pandas as pd
import matplotlib.pyplot as plt
from filtrado import ButterBandpassFilter
import numpy as np

filtro = ButterBandpassFilter(2, 40, fs=250, order=4)

# Cargar datos (ajusta ruta si es necesario)
df = pd.read_csv("subject01_session01_data.csv")

factor=(4.5/((2**23-1)*24))*16

for i in range(1,9):
    df[f"ch{i}_uV"]=df[f"ch{i}"]*factor

# Graficar los 8 canales (suponiendo nombres ch1..ch8)
time = np.linspace(0, len(df)/250, len(df))
plt.figure(figsize=(12, 6))
for i in range(1, 9):
    plt.plot(time,filtro.apply(df[f'ch{i}_uV']), label=f'Canal {i}')

plt.legend()
plt.title("Señales EMG de los 8 canales")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (µV)")
plt.savefig("emg_canales.png")
