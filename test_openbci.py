from pyOpenBCI import OpenBCICyton 

def print_raw(sample):
	print(sample.channels_data)

board=OpenBCICyton(port="/dev/ttyUSB0", daisy=False)

print("Iniciando...")
board.start_stream(print_raw)
