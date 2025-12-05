from pyOpenBCI import OpenBCICyton

def callback(sample):
    print(sample.channels_data)  # lista de 8 valores
    return  # para no bloquear

board = OpenBCICyton(port="/dev/ttyUSB0", daisy=False)
print("Streaming...")
board.start_stream(callback)
