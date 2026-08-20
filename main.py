import mediapipe as mp
import numpy as np
import cv2
import math
import json
import os
from toolbox import process_poses, stack_poses, compare_ref, absolute_to_relative

right_pose_list = [12,14,16,18,24,26,28]
right_connections = [(12,14),(14,16),(16,18),(12,24),(24,26),(26,28)]
left_pose_list = [11,13,15,17,23,25,27]
left_connections = [(11,13),(13,15),(15,17),(11,23),(23,25),(25,27)]
prev_positions_ema = dict()
line_positions_dict = dict()
pos_locations_dict = dict()
left_visibility = None
right_visibility = None
finish = False
finish_frame = None
catch = False
catch_frame = None
shoulder = None
curr_pose_list = None
curr_connections = None
first_frame = None
body_direction = 0
initial_height = 0
waited = 0
stroke_count = 0
frame_idx = 0
min_shoulder_pos = 10e99
max_shoulder_pos = -1
def get_knee_lengths(hip,knee,ankle):
    a = abs(math.dist(knee,ankle))
    b = abs(math.dist(ankle,hip))
    c = abs(math.dist(knee,hip))
    return a,b,c

def get_body_lengths(hip,shoulder):
    a = abs(math.dist(shoulder,hip))
    b = shoulder[1]-hip[1]
    return a,b

def find_angle(a,b,c):
    # finds angle using law of cosines
    right_side = (a**2 - b**2 + c**2)/(2*a*c)
    return math.acos(right_side)

def make_vert_dashed_line(dashes,GAP_MULTIPLIER,start,end,img,color,thickness):
    dist_y = (end[1]-start[1])/(dashes+(dashes-1)*GAP_MULTIPLIER)
    # cv2.line(img, hip, ghost_body_pos, (0, 255, 0), 5)
    for i in range(1, dashes+1):
        start_seg_y = round(start[1]+((i-1)*(dist_y+dist_y*GAP_MULTIPLIER)))
        end_seg_y = round(start_seg_y+dist_y)
        cv2.line(img,(start[0],start_seg_y),(start[0],end_seg_y),color,thickness)

def draw_pose_graph(img, hip, knee, ankle, shoulder_pos, ghost_body_pos, knee_angle, body_angle):
    SECTOR_RADIUS_DIVISOR = 3
    dx_kh = knee[0] - hip[0]
    dy_kh = knee[1] - hip[1]
    dx_ka = knee[0] - ankle[0]
    dy_ka = knee[1] - ankle[1]
    dx_hs = hip[0] - shoulder_pos[0]
    dy_hs = hip[1] - shoulder_pos[1]
    dx_hg = hip[0] - ghost_body_pos[0]
    dy_hg = hip[1] - ghost_body_pos[1]
    len_knee_to_hip = math.hypot(dx_kh, dy_kh)
    len_knee_to_ankle = math.hypot(dx_ka, dy_ka)
    len_hip_to_shoulder = math.hypot(dx_hs, dy_hs)
    len_hip_to_ghost = math.hypot(dx_hg, dy_hg)
    radius_k = min(len_knee_to_hip, len_knee_to_ankle) / SECTOR_RADIUS_DIVISOR
    radius_h = min(len_hip_to_shoulder, len_hip_to_ghost) / SECTOR_RADIUS_DIVISOR
    x_axis_length = abs(ankle[0] - knee[0])
    if ankle[0] < knee[0]:
        initial_angle = 180 - (math.acos(x_axis_length / len_knee_to_ankle) * (180 / math.pi))
    else:
        initial_angle = math.acos(x_axis_length / len_knee_to_ankle) * (180 / math.pi)
    INITIAL_ANGLE_OFFSET = 1.25
    ARC_ANGLE_OFFSET = 2
    overlay = img.copy()
    cv2.ellipse(
        img,
        knee,
        (round(radius_k), round(radius_k)),
        initial_angle + INITIAL_ANGLE_OFFSET,
        0,
        knee_angle - ARC_ANGLE_OFFSET,
        (160, 170, 80),
        -1,
    )
    cv2.ellipse(
        img,
        hip,
        (round(radius_h), round(radius_h)),
        -90 + INITIAL_ANGLE_OFFSET,
        0,
        body_angle - ARC_ANGLE_OFFSET,
        (160, 170, 80),
        -1,
    )
    make_vert_dashed_line(4, 2, hip, ghost_body_pos, img, (0, 255, 0), 5)
    ANGLE_INDICATOR_DISTANCE = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    (text_size_k, _) = cv2.getTextSize(f"Knee angle: {knee_angle}", font, 2, 3)
    text_w_k = text_size_k[0]
    (text_size_b, _) = cv2.getTextSize(f"Body angle: {body_angle}", font, 2, 3)
    text_w_b = text_size_b[0]
    text_pos_x_k = max(
        0,
        knee[0]
        + round(ANGLE_INDICATOR_DISTANCE * radius_k * math.cos((initial_angle + (knee_angle / 2)) * (math.pi / 180)))
        - int(text_w_k / 2),
    )
    text_pos_y_k = max(
        0,
        knee[1]
        + round(ANGLE_INDICATOR_DISTANCE * radius_k * math.sin((initial_angle + (knee_angle / 2)) * (math.pi / 180))),
    )
    text_pos_x_h = max(
        0,
        hip[0]
        + round(ANGLE_INDICATOR_DISTANCE * radius_h * math.cos(((body_angle / 2) - 90) * (math.pi / 180)))
        - int(text_w_b / 2),
    )
    text_pos_y_h = max(
        0,
        hip[1]
        + round(ANGLE_INDICATOR_DISTANCE * radius_h * math.sin(((body_angle / 2) - 90) * (math.pi / 180))),
    )
    TRANSPARENCY_ALPHA = 0.5
    cv2.addWeighted(overlay, TRANSPARENCY_ALPHA, img, 1 - TRANSPARENCY_ALPHA, 0, img)
    cv2.putText(img, f"Knee angle: {knee_angle}", (text_pos_x_k, text_pos_y_k), font, 2, (245, 245, 245), 3)
    cv2.putText(img, f"Knee angle: {knee_angle}", (text_pos_x_k + 2, text_pos_y_k + 2), font, 2, (0, 0, 0), 1)
    cv2.putText(img, f"Body angle: {body_angle}", (text_pos_x_h, text_pos_y_h), font, 2, (245, 245, 245), 3)
    cv2.putText(img, f"Body angle: {body_angle}", (text_pos_x_h + 2, text_pos_y_h + 2), font, 2, (0, 0, 0), 1)

def exponential_moving_average(cx,cy,px,py,a=0.3):
    # From testing, a=0.3 minimized knee + body angle variance
    return int(cx*a+px*(1-a)),int(cy*a+py*(1-a))

def determine_right(results,sensitivity=0.75):
    # right_pose_list = [12,14,16,18,24,26,28]
    baseline = sensitivity*6
    right_confidence_sum = 0
    left_confidence_sum = 0
    for id, lm in enumerate(results.pose_landmarks.landmark):
        if id in right_pose_list:
            right_confidence_sum += lm.visibility
        elif id in left_pose_list:
            left_confidence_sum += lm.visibility
    if max(right_confidence_sum,left_confidence_sum) < baseline:
        return -1
    if right_confidence_sum < left_confidence_sum:
        return 0
    else:
        return 1

#TODO: Refactor!!

if __name__ == '__main__':
    path = 'Videos/satvik_erg.mp4'
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    mpPose = mp.solutions.pose
    mpDraw = mp.solutions.drawing_utils
    pose = mpPose.Pose(min_detection_confidence=0.85,min_tracking_confidence=0.85)
    with open('ten_strokes.json', 'r') as file:
        strokes = json.load(file)
    ref_stroke = strokes[0][1]

    alphas = [0.2+i*0.025 for i in range(0,11)] # 10 strokes in test video
    glitches = dict()
    max_knee_glitch = -1
    max_body_glitch = -1
    prev_knee_angle = 0
    knee_angle = 0
    prev_body_angle = 0
    body_angle = 0
    a_idx = 0
    ended_stroke = False
    valid_detection = False
    FLIP_CODE = 0
    shoulder = 12
    init_frame = 0
    filler = np.ones((1024,1024,3), np.uint8) * 255
    error_message = False
    while True:
        curr_alpha = 0.275 # alphas[a_idx]
        ended_stroke = False
        success, img = cap.read()
        if not success:
            break
        if path.lower().endswith(".mov") and not valid_detection: # check this
            valid_detection = True
            img = cv2.rotate(img, cv2.ROTATE_180)
            continue
        # img = cv2.flip(img, 1)
        if FLIP_CODE:
            img = cv2.flip(img,FLIP_CODE)
            continue
        img = cv2.resize(img, (1700,1000))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            min_shoulder_pos = min(results.pose_landmarks.landmark[shoulder].x,min_shoulder_pos)
            max_shoulder_pos = max(results.pose_landmarks.landmark[shoulder].x,max_shoulder_pos)
            # Add check here (stall with continues)
            if not valid_detection and determine_right(results) == -1:
                erorr_message = True
                cv2.putText(filler, "Please move back so your body is visible", (200,512), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                cv2.imshow("placeholder", filler)
                cv2.waitKey(1)
                continue
            elif determine_right(results) == 0 and not valid_detection:
                print("ready to rumble!!")
                FLIP_CODE = 1
                valid_detection = True
                continue
            else:
                print("ready to rumble!!")
                valid_detection = True
            if error_message:
                print("attempted destruction")
                cv2.destroyWindow("placeholder")
           # mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
            h,w,c = img.shape
            for id, lm in enumerate(results.pose_landmarks.landmark):
                cx,cy = int(lm.x*w),int(lm.y*h)
                if id in prev_positions_ema:
                    px,py = prev_positions_ema[id]
                    cx,cy = exponential_moving_average(cx,cy,px,py,curr_alpha)
                line_positions_dict.update({id: (cx,cy)})
                if id in right_pose_list:
                    # print(id, lm.visibility)
                    cv2.circle(img, (cx,cy), 10, (255,0,0),cv2.FILLED)
                    if not finish:
                        if id not in pos_locations_dict.keys():
                            pos_locations_dict.update({id: [(cx,cy)]})
                        else:
                            pos_locations_dict[id].append((cx,cy))
                            if id == 12 and len(pos_locations_dict[shoulder]) >= 7:
                                # adjust this based on glitches
                                if pos_locations_dict[shoulder][-1][0] > pos_locations_dict[shoulder][-7][0]:
                                    finish_frame = frame_idx
                                    finish = True
                            elif id == 11 and len(pos_locations_dict[shoulder]) >= 7:
                                if pos_locations_dict[shoulder][-1][0] < pos_locations_dict[shoulder][-7][0]:
                                    finish_frame = frame_idx
                                    finish = True
                    elif finish and not catch:
                        pos_locations_dict[id].append((cx,cy))
                        if waited < 10 and id == shoulder: # Heuristic buffer to prevent ending stroke early
                            waited += 1
                        else:
                            if id == 12:
                                # adjust this based on glitches
                                if pos_locations_dict[shoulder][-1][0] < pos_locations_dict[shoulder][-7][0]:
                                    catch_frame = frame_idx
                                    catch = True
                            elif id == 11:
                                if pos_locations_dict[shoulder][-1][0] > pos_locations_dict[shoulder][-7][0]:
                                    catch_frame = frame_idx
                                    catch = True
                prev_positions_ema[id] = (cx,cy)
            for i,f in right_connections:
                i_cx,i_cy = line_positions_dict[i]
                f_cx,f_cy = line_positions_dict[f]
                cv2.line(img,(i_cx,i_cy),(f_cx,f_cy),(0,255,0),5)
        else:
            continue
        if catch and finish: # End of stroke
            stroke_rate = round(60*fps/((catch_frame-finish_frame) + (finish_frame-init_frame)),3)
            stroke_count += 1
            init_frame = frame_idx
            ended_stroke = True
            catch = False

            finish = False
            waited = 0
            # print(f"Distance traveled by shoulder: {round(max_shoulder_pos-min_shoulder_pos,3)*100} % of width")
            max_shoulder_pos = -1
            min_shoulder_pos = 10e99
            # print(len(pos_locations_dict[list(pos_locations_dict.keys())[0]]))
            # if stroke_count > 1:
                # compare_ref(user_list,ref_stroke)
            j_data = None
            j_path = 'user_strokes.json' #TODO: make this more robust (so it doesn't crash if poses are missing)
            if os.path.getsize(j_path) > 0:
                with open(j_path, 'r') as inputs:
                    j_data = json.load(inputs)
            else:
                with open(j_path, 'w') as output:
                    json.dump([], output)
                    j_data = []
            j_data.append(pos_locations_dict)
            with open('user_strokes.json', 'w') as output:
                json.dump(j_data, output, indent=2)
            # TODO: Add comparison function here


            pos_locations_dict = {}

            # TODO: Normalize stroke and compute similarity with avg youtube stroke
            # TODO: Compute knee and body angles on normalized stroke data


        if stroke_count:
            cv2.putText(img, f"Stroke rate: {stroke_rate}", (1200, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(img, f"Stroke count: {stroke_count}", (1200, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        h,w,c = img.shape

        VERT_SCALE = 1.2
        # ------------------Angle Calculations------------------
        hip = (line_positions_dict[24][0],line_positions_dict[24][1])
        knee = (line_positions_dict[26][0],line_positions_dict[26][1])
        ankle = (line_positions_dict[28][0],line_positions_dict[28][1])
        shoulder_pos = (line_positions_dict[12][0],line_positions_dict[12][1])
        if not first_frame:
            first_frame = True
            initial_height = round(line_positions_dict[24][1] + VERT_SCALE*(line_positions_dict[12][1]-line_positions_dict[24][1]))
        ghost_body_pos = (line_positions_dict[24][0],initial_height)
        a,b,c = get_knee_lengths(hip=hip,knee=knee,ankle=ankle)
        d,e = get_body_lengths(hip=hip,shoulder=shoulder_pos)
        knee_angle = round(find_angle(a,b,c)*(180/math.pi),1)
        if shoulder_pos[0] > hip[0]:
            body_direction = 1
        elif shoulder_pos[0] == hip[0]:
            body_direction = 0
        else:
            body_direction = -1
        body_angle = round((180-round(math.acos(e / d) * (180 / math.pi), 3)),1)*body_direction #TODO: append angles first??
        # ------------------Angular Velocity Calculations------------------
        t = 1/fps
        body_angle_v = (body_angle-prev_body_angle)/t
        knee_angle_v = (knee_angle-prev_knee_angle)/t
        cv2.putText(img, f"Body angle velocity: {round(body_angle_v)}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(img, f"Knee angle velocity: {round(knee_angle_v)}", (70, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        # ------------------GRAPHING------------------
        draw_pose_graph(img, hip, knee, ankle, shoulder_pos, ghost_body_pos, knee_angle, body_angle)
        max_knee_glitch = max(max_knee_glitch, knee_angle - prev_knee_angle)
        max_body_glitch = max(max_body_glitch, body_angle - prev_body_angle)
        if ended_stroke and stroke_count > 1: # Cycle through alphas here
            # a_idx += 1
            glitches[curr_alpha] = (round(max_knee_glitch,2),round(max_body_glitch,2))
            max_knee_glitch = 0
            max_body_glitch = 0
            # print(f"(knee,body) glitch at a={round(curr_alpha,3)}: {glitches[curr_alpha]}")
        # cv2.putText(img, f"Knee angle: {knee_angle}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        # cv2.putText(img, f"Body angle: {body_angle}", (70, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        if a_idx > 0:
            cv2.putText(img, f"Stroke variance: {glitches[alphas[a_idx - 1]]}", (70, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(img, f"Stroke alpha: {round(alphas[a_idx - 1],3)}", (70, 200), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        frame_idx += 1
        prev_knee_angle = knee_angle
        prev_body_angle = body_angle
        # 0.086 seconds per frame for left side, 0.068 for right side
