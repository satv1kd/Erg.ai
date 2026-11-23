import math
import json
import re
from dataclasses import dataclass
import numpy as np
import os
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)



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
    # finds angle using law of cosines (clamped so float error on straight legs can't raise)
    right_side = (a**2 - b**2 + c**2)/(2*a*c)
    return math.acos(max(-1.0, min(1.0, right_side)))


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
    body_angle = round((180-round(math.acos(max(-1.0, min(1.0, e / d))) * (180 / math.pi), 3)),1)*body_direction
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

def gpt_wrapper(knee_user, body_user, knee_ref, body_ref, max_knee_d, max_knee_l, max_body_d, max_body_l,
                timing=None, ref_timing=None):
    # timing / ref_timing: StrokeTiming with the detected catch and finish on the 0-100% curves.
    # Without them, assume 0% = catch and the body-angle minimum (max layback) = finish.
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
    settled_u = extrema_u[extrema_u >= 60]
    stable_u_b_finish_idx = settled_u[0] if len(settled_u) > 0 else len(body_user) - 1
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
    settled_r = extrema_r[extrema_r >= 60]
    stable_r_b_finish_idx = settled_r[0] if len(settled_r) > 0 else len(body_ref) - 1
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
    You are an experienced rowing coach providing feedback on an athlete's erg stroke. 

    Optimal stroke (slight variations are fine but user strokes should follow this):
    - Body held forward in the catch and stable during first half of drive
        - Hips and torso should move as one unit 
    - Body opens through second half of drive as hips are moving
        - Don't open the back as soon as you push with the legs
    - For the recovery reset arms first, then the body, then finally the knees
        - Arms and body must be set in recovery position before the slide moves
        - Small overlap between body and arm motion is fine, especially at higher stroke ratings
    
    
    Rules:
    - Give exactly 1-2 coaching cues, prioritized by impact
    - Be specific about timing. Use numbers to guide your response but timestamps should be qualitative
        - Halfway through the drive, start of recovery, etc
        - Mention what the user is doing wrong too, don't just recite the optimal stroke
    - Sound like a coach talking to an athlete, not a data report
    - Do not recommend they fix everything at once

    
    Timing: the catch (shoulders furthest forward) and finish (shoulders furthest back) are
    detected from the athlete's movement. Every event below is already placed relative to
    them, e.g. "halfway through the drive". Speak in those terms, not percentages.
    
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
    if timing is None:
        timing = StrokeTiming(0, user_body_finish_time, 100)
    if ref_timing is None:
        ref_timing = StrokeTiming(0, ref_body_finish_time, 100)

    # Velocities above are degrees per 1% of the curve; convert to degrees/second when the clock is known
    if timing.span_s:
        to_s = 100 / timing.span_s
        open_vel = f"{abs(round(body_u_open_vel * to_s, 1))}°/s"
        close_vel = f"{abs(round(body_u_close_vel * to_s, 1))}°/s"
    else:
        open_vel, close_vel = f"{abs(body_u_open_vel)}° per 1% of stroke", f"{abs(body_u_close_vel)}° per 1% of stroke"

    if timing.drive_s is not None:
        rhythm = (f"- Drive {timing.drive_s:.2f}s, recovery {timing.recovery_s:.2f}s "
                  f"(1:{timing.recovery_s / timing.drive_s:.1f} ratio, ~{60 / (timing.drive_s + timing.recovery_s):.0f} spm)")
    else:
        rhythm = "- Drive and recovery durations unknown"
    rhythm += (f"\n    - Drive is {timing.drive_share:.0%} of the stroke "
               f"(reference: {ref_timing.drive_share:.0%})")

    user = f"""
    Stroke rhythm:
    {rhythm}

    User metrics (reference in brackets):
    - Body starts opening {timing.describe(body_u_open_idx)} [{ref_timing.describe(body_r_open_idx)}]
    - Maximum layback {timing.describe(user_body_finish_time)} [{ref_timing.describe(ref_body_finish_time)}]
    - Body settles forward again {timing.describe(stable_u_b_finish_idx)} [{ref_timing.describe(stable_r_b_finish_idx)}]
    - Body opening speed: {open_vel}, {abs(open_percent_diff)}% {fast_or_slow_o} than reference
    - Body closing speed: {close_vel}, {abs(close_percent_diff)}% {fast_or_slow_c} than reference
    - Body angle at catch: {body_user[0]}° [{body_ref[0]}°]
    - Knee angle at catch: {knee_user[0]}° [{knee_ref[0]}°]
    - Largest body-angle gap: {round(max_body_d)}° {timing.describe(max_body_l)}
    - Largest knee-angle gap: {round(max_knee_d)}° {timing.describe(max_knee_l)}
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

# ------------------Real-time coaching helpers------------------

@dataclass
class StrokeTiming:
    """Where the catch and finish fall on a stroke's 0-100% curve (the 100-sample
    curves from stroke_angle_curves), so curve indices can be read as rowing phases.

    catch_pct is usually slightly negative: the detector only recognises a catch a few
    frames after it happens, so each recorded stroke starts just after its catch and
    ends just after the next one (next_catch_pct, a little under 100).
    """
    catch_pct: float
    finish_pct: float
    next_catch_pct: float
    span_s: float | None = None  # seconds covered by the 0-100% curve; None when unknown

    @property
    def drive_share(self):
        return (self.finish_pct - self.catch_pct) / (self.next_catch_pct - self.catch_pct)

    def _seconds(self, pct):
        return None if self.span_s is None else pct / 100 * self.span_s

    @property
    def drive_s(self):
        return self._seconds(self.finish_pct - self.catch_pct)

    @property
    def recovery_s(self):
        return self._seconds(self.next_catch_pct - self.finish_pct)

    def locate(self, pct):
        """(phase, fraction of that phase) for a curve %, e.g. ('drive', 0.5)."""
        if pct < self.finish_pct:
            phase, start, end = "drive", self.catch_pct, self.finish_pct
        else:
            phase, start, end = "recovery", self.finish_pct, self.next_catch_pct
        frac = (pct - start) / (end - start) if end > start else 0.0
        return phase, min(max(frac, 0.0), 1.0)

    def describe(self, pct):
        """Qualitative position of a curve %, e.g. 'halfway through the drive (0.41s after the catch)'."""
        phase, frac = self.locate(pct)
        for limit, words in ((0.15, "at the very start of"), (0.4, "early in"), (0.6, "halfway through"),
                             (0.85, "late in"), (1.01, "at the very end of")):
            if frac < limit:
                break
        text = f"{words} the {phase} ({frac:.0%} of the {phase}"
        seconds = self._seconds(frac * ((self.finish_pct - self.catch_pct) if phase == "drive"
                                        else (self.next_catch_pct - self.finish_pct)))
        if seconds is not None:
            text += f", {seconds:.2f}s after the {'catch' if phase == 'drive' else 'finish'}"
        return text + ")"


def average_stroke_timing(timings):
    timings = [t for t in timings if t is not None]
    if not timings:
        return None
    spans = [t.span_s for t in timings]
    return StrokeTiming(
        float(np.mean([t.catch_pct for t in timings])),
        float(np.mean([t.finish_pct for t in timings])),
        float(np.mean([t.next_catch_pct for t in timings])),
        None if None in spans else float(np.mean(spans)),
    )


def _catch_finish_indices(xs):
    # Right-facing rower: finish = shoulder furthest back (min x), catch = furthest forward
    # (max x) after the finish
    finish = int(np.argmin(xs))
    return finish, finish + int(np.argmax(xs[finish:]))


def timing_from_poses(poses, shoulder='12'):
    """StrokeTiming from frame indices alone (no clock), e.g. for reference strokes.
    Assumes the detector lag is the same at both catches, so catch_pct = next_catch_pct - 100."""
    xs = [p[0] for p in poses[shoulder]]
    if len(xs) < 2:
        return None
    finish, next_catch = _catch_finish_indices(xs)
    pct = lambda i: i / (len(xs) - 1) * 100
    return StrokeTiming(pct(next_catch) - 100, pct(finish), pct(next_catch))


@dataclass
class Stroke:
    poses: dict      # landmark id -> [(x, y), ...] for every frame of the stroke
    duration: float  # seconds
    rate: float      # strokes per minute
    travel: float    # shoulder travel as a fraction of frame width
    # Phase markers on the detector's clock (seconds)
    start_time: float = None       # first frame in poses
    end_time: float = None         # last frame in poses
    catch_time: float = None       # this stroke's catch (just before start_time)
    finish_time: float = None
    next_catch_time: float = None  # the catch that ends this stroke

    @property
    def drive_s(self):
        return self.finish_time - self.catch_time

    @property
    def recovery_s(self):
        return self.next_catch_time - self.finish_time

    def timing(self):
        span = self.end_time - self.start_time
        if span <= 0:
            return None
        pct = lambda t: (t - self.start_time) / span * 100
        return StrokeTiming(pct(self.catch_time), pct(self.finish_time), pct(self.next_catch_time), span)


class StrokeDetector:
    """main.py's finish/catch state machine, packaged for frame-by-frame use.

    Expects right-facing points keyed by right-side landmark ids (mirror left-side
    input first), so only main.py's right-side (id 12) rules are needed.
    """
    LOOKBACK = 7         # frames compared to decide shoulder direction
    SETTLE_FRAMES = 10   # heuristic buffer to prevent ending stroke early

    def __init__(self, min_travel=0.25, min_frames=15, min_rate=10, max_rate=60, shoulder=12):
        self.min_travel = min_travel  # stroke_counter.py heuristic
        self.min_frames = min_frames
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.shoulder = shoulder
        self.stroke_count = 0
        self.last_rate = None
        self.reset()

    def reset(self):
        # Drop the partial stroke; the next detected stroke starts mid-cycle so it's discarded
        self.poses = {}
        self.finish = False
        self.catch = False
        self.waited = 0
        self.start_time = None
        self.times = []                # timestamp of every frame in poses
        self.prev_catch_time = None    # catch found at the end of the previous stroke
        self.warmed_up = False

    def update(self, points, timestamp, frame_width):
        """Feed one frame of points; returns a Stroke when a valid one completes."""
        if self.start_time is None:
            self.start_time = timestamp
        self.times.append(timestamp)
        for id, pt in points.items():
            self.poses.setdefault(id, []).append(pt)
        history = self.poses[self.shoulder]
        if len(history) < self.LOOKBACK:
            return None
        if not self.finish:
            if history[-1][0] > history[-self.LOOKBACK][0]:
                self.finish = True
        elif not self.catch:
            if self.waited < self.SETTLE_FRAMES:
                self.waited += 1
            elif history[-1][0] < history[-self.LOOKBACK][0]:
                self.catch = True
        if not (self.finish and self.catch):
            return None

        # End of stroke
        poses, times, duration = self.poses, self.times, timestamp - self.start_time
        warmed_up, catch_time = self.warmed_up, self.prev_catch_time
        xs = [p[0] for p in poses[self.shoulder]]
        # The flags fire up to LOOKBACK frames late, so place the catch and finish at the
        # shoulder extremes instead. The catch ending this stroke also starts the next one.
        finish_idx, next_catch_idx = _catch_finish_indices(xs)
        self.reset()
        self.start_time = timestamp
        self.warmed_up = True
        self.prev_catch_time = times[next_catch_idx]
        travel = (max(xs) - min(xs)) / frame_width
        rate = 60 / duration if duration > 0 else 0
        if not warmed_up or travel < self.min_travel or len(xs) < self.min_frames:
            return None
        if not self.min_rate <= rate <= self.max_rate:
            return None
        self.stroke_count += 1
        self.last_rate = rate
        return Stroke(poses, duration, rate, travel, times[0], times[-1],
                      catch_time, times[finish_idx], times[next_catch_idx])


def stroke_angle_curves(stroke_poses):
    # Knee and body angles at each 1% of the stroke
    poses_x, poses_y = process_poses(stroke_poses)
    stacked = stack_poses(poses_x, poses_y)
    knee_angles, body_angles = [], []
    for i in range(len(stacked['24'])):
        knee_angle, body_angle = calc_angles(stacked['24'][i], stacked['26'][i], stacked['28'][i], stacked['12'][i])
        knee_angles.append(knee_angle)
        body_angles.append(body_angle)
    return knee_angles, body_angles


def average_angle_curves(strokes_poses):
    # Element-wise mean of each stroke's curves; strokes with degenerate geometry are skipped
    curves = []
    for poses in strokes_poses:
        try:
            curves.append(stroke_angle_curves(poses))
        except (ValueError, ZeroDivisionError, KeyError, IndexError):
            continue
    if not curves:
        return None
    knee = np.mean([c[0] for c in curves], axis=0)
    body = np.mean([c[1] for c in curves], axis=0)
    # lists, since gpt_wrapper uses list.index
    return [round(float(v), 1) for v in knee], [round(float(v), 1) for v in body]


def _load_reference_batch(path, batch_index):
    with open(path, 'r') as file:
        batches = [b for b in json.load(file) if b]
    return batches[batch_index]


def load_reference_timing(path='ten_strokes.json', batch_index=0):
    # Catch/finish positions of the reference strokes (no clock stored, so no seconds)
    timings = []
    for stroke in _load_reference_batch(path, batch_index):
        try:
            timings.append(timing_from_poses(stroke))
        except (KeyError, ValueError):
            continue
    return average_stroke_timing(timings)


def load_reference_curves(path='ten_strokes.json', batch_index=0, aspect=16/9):
    # Reference coords are normalized 0-1, so scale x by the source aspect ratio
    # before measuring angles (assumes 16:9 YouTube video)
    batch = _load_reference_batch(path, batch_index)
    scaled = [{k: [(x * aspect, y) for x, y in pts] for k, pts in stroke.items()} for stroke in batch]
    curves = average_angle_curves(scaled)
    if curves is None:
        raise ValueError(f"No usable right-side strokes in reference batch {batch_index} of {path}")
    return curves


def compare_angle_curves(knee_user, body_user, knee_ref, body_ref):
    # Same output shape as compare_ref, for curves that are already computed
    abs_knee_dist = [abs(u - r) for u, r in zip(knee_user, knee_ref)]
    abs_body_dist = [abs(u - r) for u, r in zip(body_user, body_ref)]
    max_knee_diff = max(abs_knee_dist)
    max_body_diff = max(abs_body_dist)
    return (knee_user, body_user, knee_ref, body_ref,
            max_knee_diff, abs_knee_dist.index(max_knee_diff),
            max_body_diff, abs_body_dist.index(max_body_diff))


def local_feedback(comparison, timing=None):
    # Deterministic cue used when the LLM is unavailable
    _, body_user, _, body_ref, max_knee_d, max_knee_l, max_body_d, max_body_l = comparison
    where = (lambda pct: timing.describe(pct).split(" (")[0]) if timing else (lambda pct: f"around {pct}% of the stroke")
    if max_body_d >= max_knee_d:
        cue = f"Watch your body angle {where(max_body_l)}, it's {round(max_body_d)} degrees off the reference."
    else:
        cue = f"Watch your knee angle {where(max_knee_l)}, it's {round(max_knee_d)} degrees off the reference."
    user_finish = body_user.index(min(body_user)) + 1
    ref_finish = body_ref.index(min(body_ref)) + 1
    if abs(user_finish - ref_finish) >= 5:
        timing = "early" if user_finish < ref_finish else "late"
        cue += f" You're reaching the finish {timing}, at {user_finish}% instead of {ref_finish}%."
    return cue


def strip_asterisks(text):
    # LLM replies use markdown **bold** / * bullets; drop them so cues read and speak cleanly
    text = re.sub(r"^\s*\*\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"\*+", "", text)


def coaching_feedback(comparison, use_llm=True, timing=None, ref_timing=None):
    # Returns (text, generated_by_llm); never raises so a live session can't crash on it
    # timing / ref_timing: StrokeTiming catch/finish markers for the user and reference curves
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        try:
            text = gpt_wrapper(*comparison, timing=timing, ref_timing=ref_timing)
            if text:
                return strip_asterisks(text).strip(), True
        except Exception as exc:
            print(f"LLM feedback failed, using local fallback: {exc}")
    return local_feedback(comparison, timing), False

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
    plt.plot(body2)os.e
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
