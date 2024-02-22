import cv2
from dt_apriltags import Detector
import numpy as np

visualization = True
frame_number = 0

capture = cv2.VideoCapture("/mnt/c/plugg/Examensarbete/Videor/Daaataflyg1/DJI_0082.MP4")
Camera_matrix = [[1.31754161e+03, 0.00000000e+00, 1.01639924e+03],[0.00000000e+00, 1.31547107e+03, 5.24436193e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
Camera_matrix = np.array(Camera_matrix)/2 # adjusting due to calibration being done in 1080p
camera_params = (Camera_matrix[0,0], Camera_matrix[1,1], Camera_matrix[0,2], Camera_matrix[1,2])

if (capture.isOpened()== False):  
    print("could not open video file")

at_detector = Detector(families='tag36h11',
                       nthreads=12,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

while capture.isOpened():
    frame_number += 1

    ret, frame = capture.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        tags = at_detector.detect(frame, True, camera_params, 0.130) #  130 mm är våran man mäter insidan och den anges i meter https://github.com/AprilRobotics/apriltag?tab=readme-ov-file#pose-estimation (blir 3 pixlar man mäter)
        # print(tags) # https://github.com/duckietown/lib-dt-apriltags/tree/daffy
                    # pose_t är antagligen på formen X,Y,Z och med högerhandkoordinatsystem blir det: [distans_framåt, sidled, höjd]
        for tag in tags:
            print(tag.pose_t) # prints the pose_t probably on the form X,Y,Z which translates to [distance_x, sideways_offset, alltitude_differenece]
            for idx in range(len(tag.corners)):
                cv2.line(frame, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

            cv2.putText(frame, str(tag.tag_id),
                org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255))

        if visualization:
            cv2.imshow('Detected tags', frame)
            # k = cv2.waitKey(0)

        if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
    else: 
         break 





