import sys

# Verificación inicial de NumPy
try:
    import numpy as np
    print("✅ NumPy cargado correctamente. Versión:", np.__version__)
except Exception as e:
    print("❌ Error al importar NumPy.")
    print("Detalles:", e)
    print("\nSugerencia: instala las librerías del sistema con:")
    print("   sudo apt update && sudo apt install -y libopenblas-base libopenblas-dev libatlas-base-dev")
    sys.exit(1)

# Importar pyOpenBCI
try:
    from pyOpenBCI import OpenBCICyton
except Exception as e:
    print("❌ Error al importar pyOpenBCI.")
    print("Detalles:", e)
    sys.exit(1)

# Función callback para mostrar los datos
def show_sample(sample):
    print(sample.channels_data)

# Configurar el puerto del Cyton
PORT = "/dev/ttyUSB0"  # cámbialo si tu dongle aparece como otro

print(f"Conectando al Cyton en {PORT}... (Ctrl+C para salir)")
try:
    board = OpenBCICyton(port=PORT, daisy=False)
    board.start_stream(show_sample)
except Exception as e:
    print("❌ Error al conectar con el Cyton.")
    print("Detalles:", e)
