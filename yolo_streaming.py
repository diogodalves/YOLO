import cv2
import numpy as np
import os
import imutils
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
from yolo import *
import time
# test 123
def capture_video(net, classes, output_folder, vid_name):
    vs = VideoStream(src=0).start()
    fvs = FileVideoStream(vs).start()
    time.sleep(1)
    fps = FPS().start()
    writer = None

    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(fvs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # While capturing video
    while fvs.more():
        frame = fvs.read()
        frame = imutils.resize(frame, width=1000)
        height, width = frame.shape[:2]

        outs = perform_detection(frame, net)
        indexes, boxes, class_ids, confidences = post_process_detections(outs, height, width)
        new_frame = write_detections_on_frame(indexes, boxes, class_ids, confidences, classes, frame, vid_name)
        
        # Write on video frame
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_folder+ vid_name.split('.')[0] +'_detection.avi', fourcc, 30,
                (new_frame.shape[1], new_frame.shape[0]), True)

        writer.write(new_frame)

        cv2.imshow("Frame", new_frame)

        key = cv2.waitKey(500) & 0xFF
        if key == ord("q"):
            break

        fps.update()

    writer.release()
    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':

    net, classes = import_yolo_and_classes()

    vid_name = 'test_straming.mp4'
    output_folder = 'output/'

    print('Starting Detection...')
    capture_video(net, classes, output_folder, vid_name)
    print('Done!')