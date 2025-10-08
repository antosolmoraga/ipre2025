import numpy as np
import matplotlib.pyplot as plt
from pyOpenBCI import OpenBCICyton
from collections import deque

# Parámetros de visualización
BUFFER_SIZE = 500   # muestras a mostrar en la gráfica
FS = 250            # frecuencia de muestreo del Cyton (Hz aprox.)
CHANNEL = 0         # canal a graficar (0 = primer canal)

# Buffers circulares para graficar
data_buffer = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE)
rms_buffer = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE)

# Inicializar la figura
plt.ion()
fig, ax = plt.subplots(2, 1, figsize=(10,6))
line_raw, = ax[0].plot(data_buffer, label="Señal cruda (uV)")
line_rms, = ax[1].plot(rms_buffer, label="RMS / Actividad")

ax[0].set_title("Canal EMG Cyton")
ax[0].set_ylabel("uV")
ax[1].set_ylabel("Actividad (RMS)")
ax[1].set_xlabel("Muestras")
ax[0].legend()
ax[1].legend()

# Callback de muestra
def show_sample(sample):
    # Leer un canal
    value = sample.channels_data[CHANNEL]

    # Agregar al buffer crudo
    data_buffer.append(value)

    # Calcular RMS en ventana de 50 muestras (~0.2s)
    window = np.array(list(data_buffer)[-50:])
    rms_val = np.sqrt(np.mean(window**2))
    rms_buffer.append(rms_val)

    # Actualizar gráficos
    line_raw.set_ydata(data_buffer)
    line_raw.set_xdata(np.arange(len(data_buffer)))
    line_rms.set_ydata(rms_buffer)
    line_rms.set_xdata(np.arange(len(rms_buffer)))

    ax[0].relim(); ax[0].autoscale_view()
    ax[1].relim(); ax[1].autoscale_view()
    plt.pause(0.001)

# Conexión al Cyton
print("Conectando al Cyton en /dev/ttyUSB0 ...")
board = OpenBCICyton(port="/dev/ttyUSB0", daisy=False)
board.start_stream(show_sample)
