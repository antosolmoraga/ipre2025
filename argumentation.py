import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt
from filtrado import ButterHighpassFilter, ButterBandpassFilter, NotchFilter
import matplotlib.pyplot as plt

# ------------------------------
# Parámetros generales
# ------------------------------
FS = 250
FASE_SEC = 2
SAMPLES_PER_FASE = int(0.5 * FS)
INPUT_FOLDER = "./ensayos"
OUTPUT_FILE = "./dataset_final_canal2y4.csv"


# ------------------------------
# Configurar filtros
# ------------------------------
#filtro = ButterHighpassFilter(10, fs=FS, order=4)
filtro = ButterBandpassFilter(20, 120, fs=FS, order=4)
#notch = NotchFilter(50)


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
    #mean_val = np.mean(rectified)
    #std_val = np.std(rectified) + 1e-8
    #normalized = (rectified - mean_val) / std_val
    smoothed= butter_lowpass_filter(rectified, cutoff=4, fs=FS)
    return smoothed


def add_event(df, event):
    df['etiqueta'] = -1

    for idx, row in event.iterrows():

        mask=(df['timestamp']>row['start_time'])& (df['timestamp']<=row['end_time'])
        
        if row['label'] == 'apoyo':
            df.loc[mask,'etiqueta']=1

        if row['label'] == 'oscilacion':
            df.loc[mask,'etiqueta']=0

    return df


# ------------------------------
# Procesamiento de todos los archivos
# ------------------------------
all_windows = []
factor = (4.5 / ((2**23 - 1) * 24)) * 1e6  # Conversión a microvoltios


for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".csv"):
        if filename.endswith('_events.csv'):
            continue

        filepath = os.path.join(INPUT_FOLDER, filename)
        df = pd.read_csv(filepath)
        event = pd.read_csv(filepath.replace('_data.csv','_events.csv'))


        df = add_event(df, event)

        # Convertir canal 2 a microvoltios
        df["ch2_uV"] = df["ch2"] * factor
        df["ch4_uV"] = df["ch4"] * factor
        df["ch6_uV"] = df["ch6"] * factor

        # Aplicar preprocesamiento
        df["ch2_proc"] = preprocess_emg(df["ch2_uV"])
        df["ch4_proc"] = preprocess_emg(df["ch4_uV"])
        df["ch6_proc"] = preprocess_emg(df["ch6_uV"])
        
        # eliminar primer 0.5 segundo 
        df = df.iloc[125:].reset_index(drop=True)
        
        # Crear vector de tiempo
        df["time"] = np.linspace(0, len(df) / FS, len(df))

        df['archivo']=filename
        all_windows.append(df)

        ## Graficado 
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Eje Y izquierdo: Señal EMG
        color = 'tab:blue'
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('EMG (uV)', color=color)
        ax1.plot(df['time'], df['ch4_proc'], color=color, linewidth=0.8, label='EMG Filtrado')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax1.plot(df['time'],df['ch2_proc'],color='blue',linewidth=1.5,label='Ch2 - Músculo A')
        ax1.plot(df['time'],df['ch4_proc'],color='green',linewidth=1.5,label='Ch4 - Músculo B')
        ax1.plot(df['time'],df['ch6_proc'],color='red',linewidth=1.5,label='Ch6 - Músculo C')
        ax1.legend(loc='upper left')

        # Añadir título 
        plt.title(f"Señal y Etiquetas: {filename}")
        
        # Eje Y derecho: Etiquetas (Eventos) 
        # # Instanciamos un segundo eje que comparte el mismo eje X 
        ax2 = ax1.twinx() 
        color = 'tab:orange' 
        ax2.set_ylabel('Etiqueta (Fase)', color=color) 
        
        # Usamos fill_between para que se vea como "bloques" de color de fondo 
        ax2.fill_between(df['time'], df['etiqueta'], color=color, alpha=0.3, label='Apoyo (1)') 
        ax2.tick_params(axis='y', labelcolor=color) 
        ax2.set_ylim(0, 1.5) # Fijar limite para que el bloque no tape toda la señal

        fig.tight_layout() 
        plt.show()
             
# ------------------------------
# Guardar dataset final
# ------------------------------

dataset_final = pd.concat(all_windows, ignore_index=True)
dataset_final.to_csv(OUTPUT_FILE, index=False)
print(f" Dataset final guardado en {OUTPUT_FILE} con {len(dataset_final)} filas.")

