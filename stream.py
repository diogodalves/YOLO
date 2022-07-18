from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import cv2
import torch
from PIL import Image
import os
import numpy as np
import imutils
from threading import Thread
import torchvision.models as models
from queue import Queue
import time
# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream('images/bird.mp4').start()
time.sleep(1.0)
# start the FPS timer
fps = FPS().start()

path_to_weights = 'weights/yolov5s.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_weights, verbose=False)
model.conf = 0.5
model.iou = 0.3  # NMS IoU threshold (0-1) 

while fvs.more():
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale (while still retaining 3
    # channels)
    frame = fvs.read()
    frame = imutils.resize(frame, width=450)
    
    cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()