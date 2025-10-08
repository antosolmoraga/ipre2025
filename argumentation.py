import pandas as pd
import os

# ------------------------------
# Parámetros
# ------------------------------
fs = 250           # frecuencia de muestreo (Hz)
fase_sec = 2       # duración de cada fase (s)
samples_per_fase = fs * fase_sec

input_folder = "./ensayos"        # carpeta con CSV originales
output_file = "./dataset_final.csv"  # archivo final combinado

# ------------------------------
# Crear lista para almacenar todas las ventanas
# ------------------------------
all_windows = []

# ------------------------------
# Procesar cada CSV en la carpeta ensayos
# ------------------------------
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(input_folder, filename))
        
        # Calcular número de fases completas
        num_fases = len(df) // samples_per_fase
        
        for i in range(num_fases):
            start = i * samples_per_fase
            end = start + samples_per_fase
            ventana = df.iloc[start:end].copy()
            
            # Asignar etiqueta
            if i % 2 == 0:
                ventana['etiqueta'] = 'contraccion'
            else:
                ventana['etiqueta'] = 'relajacion'
            
            # Agregar al dataset final
            all_windows.append(ventana)

# ------------------------------
# Combinar todas las ventanas
# ------------------------------
dataset_final = pd.concat(all_windows, ignore_index=True)

# Guardar CSV final
dataset_final.to_csv(output_file, index=False)
print(f"¡Dataset final guardado en {output_file} con {len(dataset_final)} filas!")
