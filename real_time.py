import numpy as np
import cv2 as cv
import pose_compat

cap = cv.VideoCapture(0)
mpPose = pose_compat
mpDraw = pose_compat
pose = mpPose.Pose(min_detection_confidence=0.85,min_tracking_confidence=0.85)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, img = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
    # Display the resulting frame
    cv.imshow('frame', img)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()