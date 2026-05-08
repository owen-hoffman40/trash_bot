import serial
import time

def main():
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

    time.sleep(2)

    ser.write(b'flashgreen\n')
    start_time = time.time()

    while time.time() - start_time < 3:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            print(line)
    time.sleep(2)

    ser.close()

if __name__ == "__main__":
	main()
