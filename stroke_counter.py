import pose_compat
import cv2
import json
import os

def exponential_moving_average(cx,cy,px,py,a=0.3):
    # From testing, a=0.3 minimized knee + body angle variance
    return int(cx*a+px*(1-a)),int(cy*a+py*(1-a))


def check_validity(path):

    # Environment setup

    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    mpPose = pose_compat
    pose = mpPose.Pose(min_detection_confidence=0.85,min_tracking_confidence=0.85)
    right_pose_list = [8,12,14,16,18,24,26,28]
    left_pose_list = [7,11,13,15,17,23,25,27]
    pos_locations_dict = dict()
    left_visibility = None
    right_visibility = None
    finish = False
    finish_frame = None
    catch = False
    catch_frame = None
    shoulder = 12
    waited = 0
    stroke_count = 0
    frame_idx = 0
    prev_stroke_rate = None
    temp = []
    max_shoulder_pos = -1
    min_shoulder_pos = 10e99
    init_frame = 0
    r_cx  = None
    r_cy = None
    r_px = None
    r_py = None
    # Loop for video processing
    while True:
        if stroke_count >= 10:  # Heuristic, 10 contiguous strokes
            ten_good_strokes = temp
            return ten_good_strokes
        success, img = cap.read()
        if not success or img is None:
            break # End of video

        # img = cv2.flip(img,-1)
        scale_width = 640
        h,w,_ = img.shape
        scale_height = int(h * scale_width / w)
        img = cv2.resize(img, (scale_width,scale_height))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks: # If landmarks are detected
            min_shoulder_pos = min(results.pose_landmarks.landmark[shoulder].x,min_shoulder_pos)
            max_shoulder_pos = max(results.pose_landmarks.landmark[shoulder].x,max_shoulder_pos)
            # Side selection based on visibility

            if left_visibility is None and right_visibility is None:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    if id == 25:
                        left_visibility = lm.visibility
                    elif id == 26:
                        right_visibility = lm.visibility
                    if left_visibility is not None and right_visibility is not None:
                        if right_visibility > left_visibility:
                            curr_pose_list = right_pose_list
                            shoulder = 12
                            break
                        elif right_visibility <= left_visibility:
                            curr_pose_list = left_pose_list
                            shoulder = 11
                            break
            visibilities = [results.pose_landmarks.landmark[lm].visibility for lm in curr_pose_list]
            if sum(visibilities) / len(curr_pose_list) < 0.85:
                stroke_count = 0
                pos_locations_dict = {}
                finish = False
                catch = False
                waited = 0
                continue

            # Determine stroke duration by logging catch and finish times
            r_px = None
            r_py = None
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h,w,_ = img.shape
                r_cx,r_cy = lm.x,lm.y
                if r_px and r_py:
                    r_cx,r_cy = exponential_moving_average(r_cx,r_cy,r_px,r_py)
                if id in curr_pose_list: # if the position is relevant

                    # Logs time that the finish is reached

                    if not finish:
                        if id not in pos_locations_dict.keys():
                            pos_locations_dict.update({id: [(r_cx,r_cy)]})
                        else:
                            pos_locations_dict[id].append((r_cx,r_cy))
                            # Right side
                            if id == 12 and len(pos_locations_dict[shoulder]) >= 7:
                                # adjust this based on glitches
                                if pos_locations_dict[shoulder][-1][0] > pos_locations_dict[shoulder][-7][0]:
                                    finish_frame = frame_idx
                                    finish = True
                            # Left side
                            elif id == 11 and len(pos_locations_dict[shoulder]) >= 7:
                                if pos_locations_dict[shoulder][-1][0] < pos_locations_dict[shoulder][-7][0]:
                                    finish_frame = frame_idx
                                    finish = True

                    # Logs time that the catch is reached (mirrored version of finish logic)

                    elif finish and not catch:
                        pos_locations_dict[id].append((r_cx,r_cy))
                        if waited < 10 and id == shoulder: # Accounts for directional change at the finish
                            waited += 1
                        else:
                            # Right side
                            if id == 12:
                                # adjust this based on glitches
                                if pos_locations_dict[shoulder][-1][0] < pos_locations_dict[shoulder][-7][0]:
                                    catch_frame = frame_idx
                                    catch = True
                            # Left side
                            elif id == 11:
                                if pos_locations_dict[shoulder][-1][0] > pos_locations_dict[shoulder][-7][0]:
                                    catch_frame = frame_idx
                                    catch = True
        if catch and finish:
            stroke_rate = round(60*fps/(catch_frame-finish_frame),3)
            shoulder_dist_traveled = round(max_shoulder_pos-min_shoulder_pos,3)
            if prev_stroke_rate is not None and abs(prev_stroke_rate-stroke_rate) <= 7 and shoulder_dist_traveled > 0.25:
                stroke_count += 1
                temp.append(pos_locations_dict)
                print(f"{stroke_count} contiguous {'stroke' if stroke_count == 1 else 'strokes'}")
            pos_locations_dict = {}
            catch = False
            finish = False
            waited = 0
            prev_stroke_rate = stroke_rate
            max_shoulder_pos = -1
            min_shoulder_pos = 10e99
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        frame_idx += 1
        r_px = r_cx
        r_py = r_cy
    return []

val = check_validity("test_dir/2000m Row in 6:40 Row Along | Real Time Tips.mp4")
j_path = "ten_strokes.json"
j_data = None
if os.path.getsize(j_path) > 0:
    with open(j_path, 'r') as inputs:
        j_data = json.load(inputs)
else:
    with open(j_path, 'w') as output:
        json.dump([], output)
        j_data = []
j_data.append(val)
with open("ten_strokes.json", 'w') as output:
    j_string = json.dump(j_data, output, indent=2)
