import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
user_input = 'on'
while user_input != 'off':
	time.sleep(1)
	user_input = input("Awaiting command: ")
	ser.write((user_input + '\n').encode('utf-8'))
	start_time = time.time()

	while time.time() - start_time < 3:
   	    if ser.in_waiting:
       		line = ser.readline().decode('utf-8').strip()
        	print(line)

ser.close()
