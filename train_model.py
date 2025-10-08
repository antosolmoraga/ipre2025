#!/usr/bin/env python3
"""
Entrenamiento de modelo EMG usando todos los CSV de la carpeta
- Protegido contra archivos faltantes o vacíos
- Extrae segmentos de contracción y relajación
- Calcula características (mean, std, rms) por canal
- Imprime cantidad de ejemplos y características
- Entrena un Random Forest y guarda el modelo
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

FS = 250  # frecuencia de muestreo

# ---------------------------
# BUSCAR TODOS LOS ARCHIVOS CSV
# ---------------------------
data_files = sorted(glob.glob("subject*_data.csv"))
all_segments = []
all_labels = []

print(f"Encontrados {len(data_files)} archivos de datos")

for data_file in data_files:
    events_file = data_file.replace("_data.csv", "_events.csv")
    
    if not os.path.exists(events_file):
        print(f"⚠️ Archivo de eventos no encontrado: {events_file}, saltando...")
        continue

    try:
        data = pd.read_csv(data_file)
        events = pd.read_csv(events_file)
    except Exception as e:
        print(f"⚠️ Error leyendo {data_file} o {events_file}: {e}, saltando...")
        continue

    if data.shape[0] == 0 or events.shape[0] == 0:
        print(f"⚠️ CSV vacío: {data_file} o {events_file}, saltando...")
        continue

    print(f"Procesando {data_file} con {events.shape[0]} eventos...")
    valid_segments = 0

    for idx, row in events.iterrows():
        try:
            # calcular índices de segmento usando iloc[0]
            start_idx = int((row['start_time'] - data['timestamp'].iloc[0]) * FS)
            end_idx   = int((row['end_time']   - data['timestamp'].iloc[0]) * FS)
            
            # proteger contra índices fuera de rango
            if start_idx >= data.shape[0] or end_idx > data.shape[0] or end_idx <= start_idx:
                print(f"⚠️ Índices fuera de rango en {data_file}, ensayo {idx+1}, saltando segmento")
                continue
            
            segment = data.iloc[start_idx:end_idx, 1:].values  # omitir timestamp
            all_segments.append(segment)
            all_labels.append(1 if row['label'] == "contraccion" else 0)
            valid_segments += 1
        except Exception as e:
            print(f"⚠️ Error procesando evento {idx+1} en {data_file}: {e}, saltando")

    print(f"✅ Segmentos válidos extraídos de {data_file}: {valid_segments}")

# ---------------------------
# EXTRAER CARACTERÍSTICAS
# ---------------------------
def extract_features(segment):
    mean = np.mean(segment, axis=0)
    std  = np.std(segment, axis=0)
    rms  = np.sqrt(np.mean(segment**2, axis=0))
    return np.concatenate([mean, std, rms])

if len(all_segments) == 0:
    print("❌ No hay segmentos válidos para entrenar. Terminar script.")
    exit()

X = np.array([extract_features(seg) for seg in all_segments])
y = np.array(all_labels)

print(f"Cantidad total de ejemplos: {X.shape[0]}, Características por ejemplo: {X.shape[1]}")

# ---------------------------
# ENTRENAR MODELO
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy del modelo: {accuracy*100:.2f}%")

# ---------------------------
# GUARDAR MODELO
# ---------------------------
joblib.dump(clf, "emg_model.pkl")
print("Modelo guardado como emg_model.pkl ✅")
