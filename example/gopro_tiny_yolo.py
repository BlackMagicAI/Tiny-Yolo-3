import numpy as np
import time
from keras.models import load_model
from PIL import Image, ImageDraw

from yolofn import yolo_out,draw

from picamera import PiCamera
from picamera.array import PiRGBArray

#########################################
# File:   gopro_tiny_yolo.py
# GoPro camera object detection using tiny-yolo 3 model.
# Author: Maurice Tedder
# Date:   August 12, 2020
##Ref:
##  https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python
##  https://picamera.readthedocs.io/en/release-1.13/recipes1.html#overlaying-images-on-the-preview
##  https://picamera.readthedocs.io/en/release-1.13/api_camera.html#module-picamera
##  https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray
##  https://github.com/raspberrypilearning/the-all-seeing-pi/blob/master/code/overlay_functions.py


######Start function definitions

# Process object detection model output and annotate image with resulting bounding
def processModelOutput(img_src, output, all_classes):
  # Process Output
  boxes, classes, scores = yolo_out(output, img_src.size)
  if boxes is not None:
    draw(img_src, boxes, scores, classes, all_classes)

#Grab data from stream and convert to numpy image data.
def convertToImage(frame, thumb_size):
  image_src = Image.fromarray(frame)
  image_thumb = image_src.resize(thumb_size, Image.BICUBIC)
  #convert to numpy array
  image = np.array(image_thumb, dtype='float32')
  image /= 255.
  image = np.expand_dims(image, axis=0)
  return image_src,image

#Used to print time lapsed during different parts of code execution.
def printTimeStamp(start, log):
  end = time.time()
  print(log + ': time: {0:.2f}s'.format(end-start))

#Combine original image & bounding box annotation overlays image into one image
# & save to jpg file.
# From Ref: https://github.com/raspberrypilearning/the-all-seeing-pi/blob/master/code/overlay_functions.py
def output_overlay(filepath, output=None, overlay=None):

  # Take an overlay Image
  overlay_img = overlay.convert('RGBA')

  # ...and a captured photo
  output_img = output.convert('RGBA')

  # Combine the two and save the image as output
  new_output = Image.alpha_composite(output_img, overlay_img)
  new_output.save(filepath, "JPEG")

######End function definitions

#Load object detection classes from text file
with open('coco_classes.txt') as f:
    class_names = f.readlines()
all_classes = [c.strip() for c in class_names]

#Load object detection model
yolo = load_model('yolomodel.h5')

#thumb size to reduce original image for tiny yolo 3 model input.
size = (416, 416)

# Create camera & rawcapture objects
#stream = BytesIO()
camera = PiCamera()
rawCapture = PiRGBArray(camera, size=camera.resolution)

#camera.start_preview(fullscreen=False, window=(0,0,size[0],size[1])) #show preview in custom size. Used for debugging.
camera.start_preview(fullscreen=True)

image_src = None
pad = None

for frame in camera.capture_continuous(rawCapture, format='rgb'):

  image_src, image = convertToImage(frame.array, size)
  orig_size = image_src.size
  
  # Create an image padded to the required size with
  # mode 'RGBA' needed for bounding box overlay with transparency mask.
  pad = Image.new('RGBA', (
      ((image_src.size[0] + (32-1)) // 32) * 32,
      ((image_src.size[1] + (16-1)) // 16) * 16,
      ))

  #Program reference start timestamp
  start = time.time()

  # Raw Output  
  output = yolo.predict(image)
  printTimeStamp(start, "Detection Time")  

  #Process model output data. annotate image with bounding boxes
  #processModelOutput(image_src, output, all_classes)
  processModelOutput(pad, output, all_classes)

  #Remove previous overlays
  for o in camera.overlays:
    camera.remove_overlay(o)
  
  o = camera.add_overlay(pad.tobytes(), alpha = 255, layer = 3, size=image_src.size)
  
  rawCapture.truncate(0)
  #break #uncomment to end loop and output image to jpg file

#Combine original image & bounding box annotation overlays image into one image
# & save to jpg file.
output_overlay("out/goproyolo.jpg", image_src, pad)

camera.stop_preview()
camera.close()






