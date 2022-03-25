import cv2
import numpy as np
import os
import imutils
from yolo import *

def capture_video(video_path, vid_name):
    vs = cv2.VideoCapture(video_path)
    writer = None

    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # While capturing video
    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        height, width = frame.shape[:2]

        go_yolo = goYOLO(frame, vid_name, output_folder, height, width)
        go_yolo.perform_detection()
        go_yolo.post_process_detections()
        new_frame = go_yolo.write_detections_on_frame()
        
        # Write on video frame
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_folder+ vid_name.split('.')[0] +'_detection.avi', fourcc, 30,
                (new_frame.shape[1], new_frame.shape[0]), True)

        writer.write(new_frame)

    writer.release()
    vs.release()

if __name__ == '__main__':
    video_folder = 'utils/'
    output_folder = 'output/'
    vid_name = os.listdir(video_folder)[0]
    video_path = video_folder + vid_name

    print('Starting Detection...')
    capture_video(video_path, vid_name)
    print('Done!')