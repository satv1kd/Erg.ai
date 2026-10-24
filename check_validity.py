from moviepy import VideoFileClip
import pose_compat
import cv2
import time
import os

folder = 'Videos'
video = os.listdir(folder)[0]
path = os.path.join(folder, video)

def isValid(path):
    mpPose = pose_compat
    pose = mpPose.Pose(min_detection_confidence=0.75,min_tracking_confidence=0.75)
    right_pose_list = [12,14,16,18,24,26,28]
    left_pose_list = [11,13,15,17,23,25,27]
    clip = VideoFileClip(path)
    increments = 1 / clip.fps
    length = clip.duration
    clips_in_middle = [clip.get_frame((length // 2) + increments * i) for i in range(0, 10)]
    valid_frames = 0
    for c in clips_in_middle:
        results = pose.process(c)
        if not results.pose_landmarks:
            return False
        landmarks = results.pose_landmarks.landmark
        right = True
        for r in right_pose_list:
            if landmarks[r].visibility < 0.75:
                right = False
                break
        left = True
        for l in left_pose_list:
            if landmarks[l].visibility < 0.75:
                left = False
                break
        if (right or left) and not (right and left):
            valid_frames += 1
    clip.close()
    return valid_frames >= 8

print(isValid(path))
