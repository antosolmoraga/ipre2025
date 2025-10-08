#!/usr/bin/env python3
"""
Adquisición EMG con OpenBCI Cyton en Raspberry Pi
- Muestra mensajes de ensayo en tiempo real
- Guarda datos de todos los canales al final
- Usa threading para que el stream no bloquee
- Agrega timeout para evitar colgado si no llegan datos
"""
import os
import time
import csv
import json
from threading import Thread
from pyOpenBCI import OpenBCICyton
import numpy as np
# source ~/emg/bin/activate

# ---------------------------
# PARÁMETROS CONFIGURABLES
# ---------------------------
PORT = "/dev/ttyUSB0"    # Ajusta según tu Raspberry Pi
FS = 250                 # Frecuencia de muestreo Cyton
N_CHANNELS = 8           # Cyton estándar
TRIALS = 3               # Número de ensayos
CONTRACT_SEC = 2         # Duración contracción
REST_SEC = 2            # Duración relajación
OUTPUT_PREFIX = "subject01_session01"
TIMEOUT_SEC = 10         # Timeout si no llegan datos (segundos)

# Buffers
data_buffer = []
events = []

# Flag para detener el thread
running = True
last_sample_time = time.time()

# ---------------------------
# CALLBACK PARA CADA MUESTRA
# ---------------------------
def handle_sample(sample):
    global last_sample_time, data_buffer
    t = time.time()

    row =[t] + sample.channels_data
    data_buffer.append(row)
    last_sample_time = time.time()

    if len(data_buffer) < 20:
        print('Starting...')

# ---------------------------
# THREAD PARA STREAM
# ---------------------------
def stream_thread(board):
    global running
    board.start_stream(handle_sample)
    
    while running:
        # Mostrar contador de muestras cada segundo
        if len(data_buffer) % FS == 0 and len(data_buffer) > 0:
            print(f"Muestras recolectadas: {len(data_buffer)}")
        # Timeout si no llegan muestras
        if time.time() - last_sample_time > TIMEOUT_SEC:
            print(f"⚠️ Timeout: no se recibieron muestras en los últimos {TIMEOUT_SEC} s, finalizando adquisición")
            running = False
            break
        time.sleep(0.01)
    board.stop_stream()
    board.ser.close()

# ---------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------
def main():
    global running, last_sample_time

    print("Conectando al Cyton...")
    board = OpenBCICyton(port=PORT, daisy=False)
    print("Esperando estabilización...")
    time.sleep(2)
    board.ser.flushInput()  # limpia bytes basura iniciales

    # Iniciar thread de adquisición
    t = Thread(target=stream_thread, args=(board,), daemon=True)
    t.start()
    
    # ---------------------------
    # PROTOCOLO DE ENSAYOS
    # ---------------------------
    for trial in range(1, TRIALS + 1):
        os.system('clear')
        print(f"\nEnsayo {trial}/{TRIALS}")
        time.sleep(2)

        # CONTRACCIÓN
        print("CONTRAE")
        start_c = time.time()
        time.sleep(CONTRACT_SEC)
        end_c = time.time()
        events.append([trial, start_c, end_c, "contraccion"])

        # RELAJACIÓN
        print("RELAJA")
        start_r = time.time()
        time.sleep(REST_SEC)
        end_r = time.time()
        events.append([trial, start_r, end_r, "relajacion"])
    # ---------------------------
    # FINALIZAR ADQUISICIÓN
    # ---------------------------
    
    running = False
    #t.join()
    print(data_buffer)
    data_buffer = np.array(data_buffer)
    print("buffer size", data_buffer.shape)
    

    # ---------------------------
    # GUARDAR CSV DE DATOS
    # ---------------------------
    data_file = f"{OUTPUT_PREFIX}_data.csv"
    with open(data_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp"] + [f"ch{i+1}" for i in range(N_CHANNELS)]
        writer.writerow(header)
        writer.writerows(data_buffer)
    print(f"Datos guardados en {data_file} (total filas: {len(data_buffer)})")

    # ---------------------------
    # GUARDAR EVENTOS
    # ---------------------------
    events_file = f"{OUTPUT_PREFIX}_events.csv"
    with open(events_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "start_time", "end_time", "label"])
        writer.writerows(events)
    print(f"Eventos guardados en {events_file}")

    # ---------------------------
    # GUARDAR METADATA
    # ---------------------------
    metadata = {
        "subject": OUTPUT_PREFIX.split("_")[0],
        "session": OUTPUT_PREFIX.split("_")[1],
        "fs": FS,
        "n_channels": N_CHANNELS,
        "trials": TRIALS,
        "contract_sec": CONTRACT_SEC,
        "rest_sec": REST_SEC,
        "port": PORT,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    meta_file = f"{OUTPUT_PREFIX}_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata guardada en {meta_file}")

if __name__ == "__main__":
    main()
