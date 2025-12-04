import os
import time
import csv
import json
from threading import Thread
from pyOpenBCI import OpenBCICyton
import numpy as np

# ---------------------------
# PARÁMETROS CONFIGURABLES
# ---------------------------
PORT = "/dev/ttyUSB0"    # Ajusta según tu Raspberry Pi
FS = 250                 # Frecuencia de muestreo Cyton
N_CHANNELS = 8           # Cyton estándar
TRIALS = 30               # Número de ensayos
PASO_DER = 1             # Duración paso derech
PASO_IZQ = 1             # Duración paso izquierdo
OUTPUT_PREFIX = "subject01_session27"
TIMEOUT_SEC = 10         # Timeout si no llegan datos (segundos)

#### LOS ELECTRODOS VAN EN LA PIERNA DERECHA 

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
    global last_sample_time
    t = time.time()
    row = [t] + sample.channels_data[:N_CHANNELS]
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

	#print(f"\nEnsayo {trial}/{TRIALS}")
        #time.sleep(2)

        # PASO CON PIERNA DERECHA
        print("PIE DERECHO")
        start_c = time.time()
        time.sleep(PASO_DER)
        end_c = time.time()
        events.append([trial, start_c, end_c, "apoyo"])

        # PASO CON PIERNA IZQUIERDA
        print("PIE IZQUIERDO")
        start_r = time.time()
        time.sleep(PASO_IZQ)
        end_r = time.time()
        events.append([trial, start_r, end_r, "oscilacion"])

    # ---------------------------
    # FINALIZAR ADQUISICIÓN
    # ---------------------------
    
    running = False
    #t.join()
    print(data_buffer)
    data_buffer_a = np.array(data_buffer)
    print(data_buffer_a.shape)
    
    #np.save()

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
        "apoyo": PASO_DER,
        "oscilacion": PASO_IZQ,
        "port": PORT,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    meta_file = f"{OUTPUT_PREFIX}_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata guardada en {meta_file}")

if __name__ == "__main__":
    main()

