import time
import picamera

#Test program for video camera input on the raspberry Pi with screen preview.
camera = picamera.PiCamera()

try:
    camera.start_preview()
    time.sleep(10)
    camera.stop_preview()
finally:
    camera.close()
