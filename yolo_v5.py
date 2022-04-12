import cv2
import torch
from PIL import Image
import os
import numpy as np
import imutils
import logging

class goYOLOv5:
    def __init__(self, file_directory, file_name, file_path):
        self.file_directory = file_directory
        self.file_name = file_name
        self.file_path = file_path

    def get_file(self):
        if self.file_name.split('.')[1] in ['jpg', 'jpeg', 'png']:
            img_frame = Image.open(file_path)
            return img_frame

        else:
            self.file = cv2.VideoCapture(file_path)
            return self

    def yolov5(self, img_frame):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', verbose=False)
        model.conf = 0.6

        # Inference
        results = model([img_frame], size=640)

        self.points = results.pandas().xyxy[0].iloc[:, 0:4].values
        self.classes = results.pandas().xyxy[0]['name'].values
        self.confidences = results.pandas().xyxy[0]['confidence'].values
        self.class_ids = results.pandas().xyxy[0]['class'].astype('category').cat.codes
        self.img_frame = img_frame

        return self

    def contour_detections(self):

        contour_size = 2

        self.new_frame = cv2.cvtColor(np.array(self.img_frame), cv2.COLOR_BGR2RGB)
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        font = cv2.FONT_HERSHEY_PLAIN

        # Compute contours
        for i in range(len(self.points)):
            x_min, y_min, x_max, y_max = self.points[i]
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            label = str(self.classes[self.class_ids[i]])
            color = [int(c) for c in colors[self.class_ids[i]]]
            confidence = str(round(self.confidences[i], 3))

            cv2.rectangle(self.new_frame, (x_min, y_max), (x_max, y_min), color, contour_size)

            cv2.rectangle(self.new_frame, (x_min, y_min), 
                    (x_min+250, y_min-50), 
                    color, -1)
                    
            cv2.putText(self.new_frame, '{} {}'.format(label, confidence), (x_min, y_min - 10), 
                    font, 
                    contour_size, 
                    (255,255,255), 
                    contour_size)

        return self

    def write_image_on_directory(self):
        cv2.imwrite(self.file_directory + '/' + '{}_detection.jpg'.format(self.file_name.split('.')[0]), self.new_frame)
        
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

        # While capturing video
        while True:
            (grabbed, frame) = self.file.read()

            if not grabbed:
                break
            height, width = frame.shape[:2]

            frame = Image.fromarray(frame)

            i_choose_yo_lo.yolov5(frame)
            i_choose_yo_lo.contour_detections()
                    
            # Write on video frame
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output_folder+ self.file_name.split('.')[0] +'_detection.avi', fourcc, 30,
                    (self.new_frame.shape[1], self.new_frame.shape[0]), True)

            writer.write(self.new_frame)

        writer.release()
        self.file.release()

if __name__ == '__main__':
    file_folder = 'utils/'
    output_folder = 'output/'
    file_name = os.listdir(file_folder)[0]
    file_path = file_folder + file_name

    print('Detecting...')

    # Images
    i_choose_yo_lo = goYOLOv5(output_folder, file_name, file_path)
    image_frame = i_choose_yo_lo.get_file()
    i_choose_yo_lo.yolov5(image_frame)
    i_choose_yo_lo.contour_detections()
    i_choose_yo_lo.write_image_on_directory()

    # Videos
    # i_choose_yo_lo = goYOLOv5(output_folder, file_name, file_path)
    # i_choose_yo_lo.get_file()
    # i_choose_yo_lo.capture_video()