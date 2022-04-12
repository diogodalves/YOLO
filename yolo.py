import cv2
from cv2 import COLOR_HSV2RGB_FULL
import numpy as np
import os

class goYOLO:
    def __init__(self, img, img_name, output_folder, height, width):
        self.img = img
        self.output_folder = output_folder
        self.img_name = img_name
        self.height = height
        self.width = width
        
        self.net = cv2.dnn.readNet("yolo-coco/yolov3.weights", "yolo-coco/yolov3.cfg")

        self.classes = []
        with open("yolo-coco/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        f.close()

    # Scale/Resize/Channels/Grayscale and don't crop
    def perform_detection(self):
        blob = cv2.dnn.blobFromImage(self.img, 1/255, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()
        self.outs = self.net.forward(layer_names)
        
    # Showing informations on the screen
    def post_process_detections(self):
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        for out in self.outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence >= 0.75:
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    self.boxes.append([x, y, w, h])
                    self.confidences.append(float(confidence))
                    self.class_ids.append(class_id)

        self.indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)

    def write_detections_on_frame(self):
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(self.boxes)):
            if i in self.indexes:
                x, y, w, h = self.boxes[i]
                label = str(self.classes[self.class_ids[i]])
                confi = str(round(self.confidences[i], 2))
                color = [int(c) for c in colors[self.class_ids[i]]]
                cv2.rectangle(self.img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(self.img, label+''+confi, (x, y - 10), font, 2, color, 2)

        img_types = ['jpg', 'jpeg', 'png']
        for img_type in img_types:
            if self.img_name.split('.')[1] == img_type:
                cv2.imwrite(self.output_folder+ self.img_name.split('.')[0] +'_detection.jpg', self.img)
        
        return self.img

if __name__ == '__main__':
    img_folder = 'utils/'
    output_folder = 'output/'
    img_name = os.listdir(img_folder)[0]
    image_path = img_folder + img_name
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    print('Detecting...')

    go_yolo = goYOLO(img, img_name, output_folder, height, width)
    go_yolo.perform_detection()
    go_yolo.post_process_detections()
    go_yolo.write_detections_on_frame()

    print('Done!')