
# Examples included in this project
### yolomain.py
This example reads an image file from the images folder, performs object detection on it and saves the result image annotated with bounding boxes into the out folder.
### camtest.py
This example program is intended to be run on a Raspberry Pi. It is a test program for video camera input on the raspberry Pi with screen preview. It will display a full screen 10 second preview of live camera video for the camera connected to the CSI-2 input of a Raspberry Pi.
The camera video can also be tested by running the following command from the command line:

    raspvid -t 0 -md 1

Ref: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119415572.app2

### gopro_tiny_yolo.py
This example demonstrates running real time (almost!) Tiny-Yolo 3 object detection on video streaming from the camera connected to the CSI-2 input of a Raspberry Pi. It was tested using the following hardware:
 - Auvidea 70501 HDMI to CSI-2 bridge
 - GoPro Hero 7 Black
 - Insignia - Micro HDMI to HDMI Adapter - Black
	 - Model: NS-HG1182
	 - SKU: 6167620
 - Raspberry Pi 3 Model B+1.4 GHz Cortex-A53 with 1 GB Ram
	 - TensorFlow 1.11.0 installed
	 - Keras 2.2.4 installed
	 - Python 3.5.3 installed

> Written with [StackEdit](https://stackedit.io/).
