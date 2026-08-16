import math
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()



#TODO: eventually move all helper functions here


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


def calc_angles(hip,knee,ankle,shoulder):
    a,b,c = get_knee_lengths(hip=hip,knee=knee,ankle=ankle)
    d,e = get_body_lengths(hip=hip,shoulder=shoulder)
    knee_angle = round(find_angle(a,b,c)*(180/math.pi),1)
    if shoulder[0] > hip[0]:
        body_direction = 1
    elif shoulder[0] == hip[0]:
        body_direction = 0
    else:
        body_direction = -1
    body_angle = round((180-round(math.acos(e / d) * (180 / math.pi), 3)),1)*body_direction
    return knee_angle,body_angle

# [12,14,16,18,24,26,28]

#TODO: DEBUG ABSOLUTE TO RELATIVE!!!

def absolute_to_relative(poses_list, width, height):
    new_pose_list = {}
    for p in poses_list:
        new_pose_list[p] = []
        for i in range(len(poses_list[p])):
            new_pose_list[p].append((poses_list[p][i][0] / width, poses_list[p][i][1] / height))
    return new_pose_list

def normalize_t(y_old):
    x_old = np.linspace(0, 1, len(y_old))
    x_new = np.linspace(0, 1, 100)
    return np.interp(x_new, x_old, y_old)

def process_poses(poses_list): # Used for graphing purposes
    global poses_x
    global poses_y
    pose_list = []
    if type(list(poses_list.keys())[0]) == int:
        pose_list = [12,14,16,18,24,26,28]
    else:
        pose_list = ['12','14','16','18','24','26','28']
    poses_x = [[i[0] for i in poses_list[p]] for p in pose_list]
    for i in range(len(poses_x)):
        poses_x[i] = normalize_t(poses_x[i])
    poses_y = [[i[1] for i in poses_list[p]] for p in pose_list]
    for i in range(len(poses_y)):
        poses_y[i] = normalize_t(poses_y[i])
    return poses_x, poses_y # list of numpy arrays

def stack_poses(poses_x,poses_y):
    output = dict()
    idx = 0
    for k in ['12','14','16','18','24','26','28']:
        output[k] = []
        for i in range(len(poses_x[0])):
            output[k].append((poses_x[idx][i], poses_y[idx][i]))
        idx += 1
    return output

def compare_ref(pose_list, ref_list):
    # For inputs, both are normalized to len(ref_list['24']) points
    
    # angle diffs during % of stroke + max diffs

    p_hip = pose_list['24']
    p_knee = pose_list['26']
    p_ankle = pose_list['28']
    p_shoulder = pose_list['12']
    r_hip = ref_list['24']
    r_knee = ref_list['26']
    r_ankle = ref_list['28']
    r_shoulder = ref_list['12']
    p_knee_angles = []
    p_body_angles = []
    r_knee_angles = []
    r_body_angles = []
    abs_knee_dist = []
    abs_body_dist = []
    print(p_hip)
    for i in range(len(ref_list['24'])):
        p_knee_angle, p_body_angle = calc_angles(p_hip[i],p_knee[i],p_ankle[i],p_shoulder[i])
        p_knee_angles.append(p_knee_angle)
        p_body_angles.append(p_body_angle)
        r_knee_angle, r_body_angle = calc_angles(r_hip[i],r_knee[i],r_ankle[i],r_shoulder[i])
        r_knee_angles.append(r_knee_angle)
        r_body_angles.append(r_body_angle)
        abs_knee_dist.append(abs(p_knee_angle - r_knee_angle))
        abs_body_dist.append(abs(p_body_angle - r_body_angle))
    max_knee_diff = max(abs_knee_dist)
    max_knee_loc = abs_knee_dist.index(max_knee_diff)
    max_body_diff = max(abs_body_dist)
    max_body_loc = abs_body_dist.index(max_body_diff)
    #TODO: Add rowing phase segmentation using slide movement
    print(f"Peak knee angle deviation of {round(max_knee_diff)} degrees at {max_knee_loc}% of the stroke")
    print(f"Peak body angle deviation of {round(max_body_diff)} degrees at {max_body_loc}% of the stroke")
    # overall stroke variation
    knee_rmse = math.sqrt(sum(((abs_knee_dist[j])**2) for j in range(len(ref_list['24']))) / len(ref_list['24']))
    body_rmse = math.sqrt(sum(((abs_body_dist[j])**2) for j in range(len(ref_list['24']))) / len(ref_list['24']))
    print(f"Knee angle deviates {round(knee_rmse)}% from the reference stroke on average")
    print(f"Body angle deviates {round(body_rmse)}% from the reference stroke on average")
    return p_knee_angles, p_body_angles, r_knee_angles, r_body_angles, max_knee_diff, max_knee_loc, max_body_diff, max_body_loc

def gpt_wrapper(knee_user, body_user, knee_ref, body_ref, max_knee_d, max_knee_l, max_body_d, max_body_l):
    # Initialization
    user_body_finish_angle = min(body_user)
    user_body_finish_time = body_user.index(user_body_finish_angle) + 1 # 1 indexed (1-100%)
    ref_body_finish_angle = min(body_ref)
    ref_body_finish_time = body_ref.index(ref_body_finish_angle) + 1
    diff = user_body_finish_angle - ref_body_finish_angle # -diff: body farther back, +diff: body farther forward
    THRESHOLD = 0.2  # tune as necessary

    # User calculations
    body_u_gradient = np.gradient(body_user)
    extrema_u = np.where(abs(body_u_gradient) < THRESHOLD)[0]
    candidates_u = extrema_u[extrema_u <= 25]
    stable_u_b_finish_idx = extrema_u[extrema_u >= 60][0]
    body_u_open_idx = candidates_u[-1] if len(candidates_u) > 0 else 0
    body_u_open_vel = round((user_body_finish_angle - body_user[body_u_open_idx]) / (user_body_finish_time - body_u_open_idx), 2)
    body_u_close_vel = round((body_user[stable_u_b_finish_idx] - user_body_finish_angle) / (stable_u_b_finish_idx - user_body_finish_time), 2)

    # Ref calculations
    body_r_gradient = np.gradient(body_ref)
    # plt.plot(body_ref)
    # plt.plot(body_user)
    # plt.plot(body_r_gradient)
    # plt.show()
    extrema_r = np.where(abs(body_r_gradient) < THRESHOLD)[0]
    candidates_r = extrema_r[extrema_r <= 25]
    stable_r_b_finish_idx = extrema_r[extrema_r >= 60][0]
    body_r_open_idx = candidates_r[-1] if len(candidates_r) > 0 else 0
    body_r_open_vel = round((ref_body_finish_angle - body_ref[body_r_open_idx]) / (ref_body_finish_time - body_r_open_idx), 2)
    body_r_close_vel = round((body_ref[stable_r_b_finish_idx] - ref_body_finish_angle) / (stable_r_b_finish_idx - ref_body_finish_time), 2)

    # Comparisons
    print(body_u_open_vel, body_r_open_vel)
    print(body_u_close_vel, body_r_close_vel)
    open_percent_diff = round(((body_u_open_vel / body_r_open_vel) - 1) * 100) # format: x% faster than ref
    close_percent_diff = round(((body_u_close_vel / body_r_close_vel) - 1) * 100)
    fast_or_slow_o = "faster" if open_percent_diff > 0 else "slower"
    fast_or_slow_c = "faster" if close_percent_diff > 0 else "slower"

    system = """
    You are an experienced rowing coach analyzing an athlete's erg stroke 
    compared to an elite reference stroke. 
    
    Rules:
    - Give exactly 1-2 coaching cues, prioritized by impact
    - Be specific about timing and numbers
    - Sound like a coach talking to an athlete, not a data report
    - Do not recommend they fix everything at once
    
    Important: 0-100% represents the ENTIRE stroke (0% = catch, 100% = return to catch)
    Body should ideally start opening at the middle of the drive, so around 20-30%.
    
    """
    '''
    (reference: {body_r_open_idx}%)
    (reference: {ref_body_finish_time}%)
    (reference: {body_r_open_vel}°/s, user is {abs(open_percent_diff)}% {fast_or_slow_o})
    (reference: {body_r_close_vel}°/s, user is {abs(close_percent_diff)}% {fast_or_slow_c})
    (reference: {body_ref[0]}°)
    (reference: {knee_ref[0]}°)
    Peak body angle deviation: {max_body_d}° at {max_body_l}% of the stroke
    Peak knee angle deviation: {max_knee_d}° at {max_knee_l}% of the stroke 
    '''
    #TODO: add timestamps for stroke segments here
    user = f"""
    User metrics:
    - Body opening initiates at: {body_u_open_idx}% of stroke 
    - User finish location: {user_body_finish_time}% of stroke 
    - Body opening average velocity from start to finish: {body_u_open_vel}°/s 
    - Body closing average velocity from start to finish: {body_u_close_vel}°/s 
    - Body angle at catch: {body_user[0]}° 
    - Knee angle at catch: {knee_user[0]}° 
     """

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    feedback = response.choices[0].message.content

    return feedback

if __name__ == '__main__':

    with open('ten_strokes.json', 'r') as file:
        strokes = json.load(file)

    with open('user_strokes.json', 'r') as file:
        user_strokes = json.load(file)

    list1 = strokes[0][3]

    list2 = user_strokes[3]

    p_x, p_y = process_poses(list1)
    list1 = stack_poses(p_x,p_y)
    p_x, p_y = process_poses(list2)
    list2 = stack_poses(p_x,p_y)

    knee_u, body_u, knee_r, body_r, knee_d, knee_l, body_d, body_l = compare_ref(list2, list1)
    feedback = gpt_wrapper(knee_u,body_u,knee_r,body_r, knee_d, knee_l, body_d, body_l)

    print(feedback)
    '''
    # plt.plot(body1)
    plt.plot(body2)
    plt.show()
    plt.plot(knee1)
    plt.plot(knee2)
    plt.show() '''

''' 
Metrics to calculate:
- Body angle magnitude and index at finish
    - Distance off reference stroke
- Forward and backward body angle velocities
- Peak knee angle timing

'''

'''
You're opening a bit too early — hold the torso angle until about **20–25%** of the drive, then swing through. Right now you start at **11%**, so let the legs do more of the work before you pick the body up.

Also, your finish is **short at 41%** of the stroke. Keep the handle moving to a **full finish around 45–50%**, with a clean layback and strong draw to the body.

'''


'''
Alright, listen up. First off, we need to work on your body opening timing.
Right now, you're initiating it too early at 11%, while the elite reference kicks in right at the catch. 
Focus on keeping your body closed until you hit the catch—let's see if you can delay that body opening until 5% of the stroke for starters.

Second, let’s address your finish position. You're coming through at 41%, but the ideal is 33%. 
This is too far back and can lead to inefficiency. 
Keep your finish tight and drive your hands away from your body just a bit quicker as you hit the release. 
Let’s work on these two points, and we'll build from there!
'''
