#Tiny Yolo 3 model test program
#BlackMagicai.com
import numpy as np
import time
from keras.models import load_model
from PIL import Image, ImageDraw

from yolofn import yolo_out,draw

yolo = load_model('yolotest.h5')

#########################################
#Load input image from images folder add resize to match yolo model imput size
input_image_name = 'dog.jpg'

size = (416, 416)
image_src = Image.open("images/" + input_image_name)
orig_size = image_src.size
image_thumb = image_src.resize(size, Image.BICUBIC)
image = np.array(image_thumb, dtype='float32')
image /= 255.
image = np.expand_dims(image, axis=0)

##############################

# Raw Output
start = time.time() #prediction start time
output = yolo.predict(image)
end = time.time() #prediction end time

print('Processing time: {0:.2f}s'.format(end-start))

############################

with open('coco_classes.txt') as f:
    class_names = f.readlines()
all_classes = [c.strip() for c in class_names]

# Processed Output
thumb_size = image_thumb.size
boxes, classes, scores = yolo_out(output, thumb_size)
if boxes is not None:
  draw(image_thumb, boxes, scores, classes, all_classes)

# Display processed image output
# image_src.show()  
image_thumb.save("out/" + input_image_name, "JPEG")

