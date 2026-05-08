import serial
import time

def main():
    ser = serial.Serial(
        port='/dev/ttyACM0',
        baudrate=9600,
        timeout=1,
        dsrdtr=False,
        rtscts=False
    )
    ser.dtr = False
    ser.rts = False

    time.sleep(2)

    ser.write(b'180\n')
    start_time = time.time()

    while time.time() - start_time < 3:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            print(line)
    time.sleep(2)

    ser.close()

if __name__ == "__main__":
    main()

