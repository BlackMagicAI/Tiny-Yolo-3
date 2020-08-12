import numpy as np
import time
from keras.models import load_model
from PIL import Image, ImageDraw

from yolofn import yolo_out,draw
from io import BytesIO
from picamera import PiCamera

#########################################

# Process object detection model output and annotate image with resulting bounding
def processModelOutput(img_src, output, all_classes):
  # Process Output
  boxes, classes, scores = yolo_out(output, img_src.size)
  if boxes is not None:
    draw(img_src, boxes, scores, classes, all_classes)

#Grab data from stream and convert to numpy image data.
#Return original image and image as numpy array.
def convertToImage(stream):
  image_src = Image.open(stream)
  image_thumb = image_src.resize(size, Image.BICUBIC)
  image = np.array(image_thumb, dtype='float32')
  image /= 255.
  image = np.expand_dims(image, axis=0)
  return image_src,image

#Load object detection classes from text file
with open('coco_classes.txt') as f:
    class_names = f.readlines()
all_classes = [c.strip() for c in class_names]

#Load object detection model
yolo = load_model('yolomodel.h5')

size = (416, 416)

# Create the in-memory stream
stream = BytesIO()
camera = PiCamera()
camera.start_preview(fullscreen=False, window=(0,0, size[0], size[1]))
#camera.start_preview(fullscreen=True) #uncomment for full screen preview

camera.capture(stream, "jpeg")
# Truncate the stream to the current position (in case
# prior iterations output a longer image)
stream.truncate()
stream.seek(0)

#grad image from camera stream
image_src, image = convertToImage(stream)

##############################

# Raw Output from yolo prediction
start = time.time()
output = yolo.predict(image)
end = time.time()

print('time: {0:.2f}s'.format(end-start))

#Process model output data. annotate original image with bounding boxes
processModelOutput(image_src, output, all_classes)

#time.sleep(5)
 
image_src.save("out/gopro.jpg", "JPEG")

camera.stop_preview()
camera.close()






