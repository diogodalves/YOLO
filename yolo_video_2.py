import cv2
import numpy as np
import os
import imutils
from yolo import *
from threading import Thread
import time

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.capture.read()
            writer = None

            if not self.grabbed:
                break

            self.height, self.width = self.frame.shape[:2]

            net, classes = import_yolo_and_classes()

            outs = perform_detection(self.frame, net)

            indexes, boxes, class_ids, confidences = post_process_detections(outs, self.height, self.width)

            self.new_frame = write_detections_on_frame(indexes, boxes, class_ids, confidences, classes, self.frame, vid_name)
            
            # Write on video frame
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output_folder+ vid_name.split('.')[0] +'_detection.avi', fourcc, 30,
                    (self.new_frame.shape[1], self.new_frame.shape[0]), True)

            writer.write(self.new_frame)

        writer.release()
        time.sleep(self.FPS)

    def show_frame(self):
        cv2.imshow('frame', self.new_frame)
        cv2.waitKey(self.FPS_MS)

if __name__ == '__main__':
    video_folder = 'utils/'
    output_folder = 'output/'
    vid_name = os.listdir(video_folder)[0]
    video_path = video_folder + vid_name
 
    print('Starting Detection...')

    threaded_camera = ThreadedCamera(video_path)
    while True:
        try:
            threaded_camera.show_frame()
        except AttributeError:
            pass