import sensor
import image
import time
import pyb

# Camera setup
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.VGA)
sensor.skip_frames(time=2000)

# exposure
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)

usb = pyb.USB_VCP()

while True:
    if usb.any():
        command = usb.readline().decode().strip()

        if command == "CAPTURE":
            img = sensor.snapshot()

            # Save image to OpenMV board
            img.save("capture.jpg", quality=90)

            f = open("capture.jpg", "rb")
            data = f.read()
            f.close()

            usb.write("START")
            usb.write(str(len(data)))
            usb.write(data)

            usb.write("END")