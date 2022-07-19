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
from imutils.video import FileVideoStream
from imutils.video import FPS

class goYOLOv5:
    def __init__(self, file_path, output_directory, path_to_weights):
        self.output_directory = output_directory
        self.file_path = file_path
        self.path_to_weights = path_to_weights

    def get_file(self):
        if self.file_path.split('/')[1].split('.')[1] in ['jpg', 'jpeg', 'png']:
            img_frame = Image.open(self.file_path)
            return img_frame

        else:
            self.file = cv2.VideoCapture(file_path)
            return self

    def load_model(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.path_to_weights, verbose=False)
        self.model.conf = 0.5
        self.iou = 0.3

    def yolov5_results(self, img_frame):

        # Inference
        results = self.model([img_frame], size=640)

        self.points = results.pandas().xyxy[0].iloc[:, 0:4].values
        self.classes = results.pandas().xyxy[0]['name'].values
        self.confidences = results.pandas().xyxy[0]['confidence'].values
        self.class_ids = results.pandas().xyxy[0]['class'].astype('category').cat.codes
        self.img_frame = img_frame

        return self

    def contour_detections(self):

        alpha = 0.68
        contour_size = 3
        vertices = []
        contours = []
        self.new_frame = cv2.cvtColor(np.array(self.img_frame), cv2.COLOR_BGR2RGB)
        
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        font = cv2.FONT_HERSHEY_PLAIN

        # Compute contours
        for i in range(len(self.points)):
            overlay = self.new_frame.copy()
            x_min, y_min, x_max, y_max = self.points[i]
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            vertices = [x_min, y_min, x_max, y_max]
            contours.append(vertices)

            label = str(self.classes[self.class_ids[i]])

            # For multiple classes
            color = [int(c) for c in colors[self.class_ids[i]]]

            # For a single class
            # color = [3, 3, 186]

            # confidence = str(round(self.confidences[i], 3))

            cv2.rectangle(overlay, (x_min, y_max), (x_max, y_min), color, contour_size)
            cv2.rectangle(overlay, (x_min, y_min), (x_min+250, y_min-50), color, -1)
            cv2.putText(overlay, '{}'.format(label), (x_min, y_min - 5), font, contour_size, (255,255,255), contour_size)
            cv2.addWeighted(overlay, alpha, self.new_frame, 1 - alpha, 0, self.new_frame)

        return self, contours

    def write_image_on_directory(self):
        cv2.imwrite(self.output_directory + '/' + '{}_detection_yolo.jpg'.format(self.file_path.split('/')[1].split('.')[0]), self.new_frame)

        return self.new_frame
        
    def capture_video(self):
        writer = None

        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
            total = int(self.file.get(prop))
            print("[INFO] {} total frames in video".format(total))

        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total = -1

        while True:
            (grabbed, frame) = self.file.read()

            if not grabbed:
                break
            height, width = frame.shape[:2]

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = Image.fromarray(frame)

            self.yolov5_results(frame)
            self.contour_detections()

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(self.output_directory + self.file_path.split('/')[1].split('.')[0] +'_detection.avi', 
                fourcc, 30, (self.new_frame.shape[1], self.new_frame.shape[0]), True)

            writer.write(self.new_frame)

        writer.release()
        self.file.release()

    def run_yolo_on_stream(self):
        self.load_model()
        writer=None

        if self.file_path.split('/')[1] == 'webcam':
            path = 0
            file_name = 'webcam.mp4'
        else:
            path = self.file_path
            file_name = self.file_path.split('/')[1]

        fvs = FileVideoStream(path, transform=None, queue_size=4).start()
        time.sleep(1.5)
        fps = FPS().start()

        while fvs.more():
            frame = fvs.read()
            frame = imutils.resize(frame, width=700)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            self.yolov5_results(frame)
            self.contour_detections()

            # Write on video frame
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(self.output_directory + file_name +'_detection.avi', fourcc, 30,
                    (self.new_frame.shape[1], self.new_frame.shape[0]), True)

            writer.write(self.new_frame)
    
            cv2.imshow("Frame", self.new_frame)
            cv2.waitKey(1)
            fps.update()

        writer.release()
        self.file.release()
        fps.stop()
        cv2.destroyAllWindows()
        fvs.stop()

    def run_yolo_on_images(self):
        self.image_frame = self.get_file()
        self.load_model()
        self.yolov5_results(self.image_frame)
        self.contour_detections()
        self.write_image_on_directory()

    def run_yolo_on_videos(self):
        self.load_model()
        self.get_file()
        self.capture_video()

if __name__ == '__main__':
    file_folder = 'images/'
    output_folder = 'output/'
    file_name = 'webcam'
    file_path = file_folder + file_name
    path_to_weights = 'weights/yolov5n.pt'

    print('Detecting...')

    # Images
    # goYOLOv5(file_path, output_folder, path_to_weights).run_yolo_on_images()

    # Videos
    # goYOLOv5(file_path, output_folder, path_to_weights).run_yolo_on_videos()

    # Streaming
    goYOLOv5(file_path, output_folder, path_to_weights).run_yolo_on_stream()

    print('Finished!')
