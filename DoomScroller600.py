"""DoomScroller600 webcam eyebrow scrolling controller.

This is the user's supplied Scroller1200.py baseline, installed under the
project's canonical filename. See README.md for controls and known limitations.
"""

VERSION = "0.5P"
import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque

# ---------- Settings ----------
SCROLL_AMOUNT = 800

# Calibration
CALIBRATION_FRAMES = 60      # Frames to collect for baseline
BASELINE_BUFFER_SIZE = 60    # Smoothing for baseline (must be >= CALIBRATION_FRAMES)

# Gesture detection
GESTURE_HOLD_TIME = 0.08    # Must hold gesture this long (balanced for reliability)
GESTURE_MAX_DURATION = 0.75  # Maximum gesture duration (prevents long holds from triggering)
GESTURE_COOLDOWN = 0.3       # Cooldown after triggering (reduced to prevent getting stuck)
THRESHOLD_MULTIPLIER = 1.04  # Lower threshold for easier triggering
HYSTERESIS_PERCENT = 0.08    # 8% margin for clear state detection (increased hysteresis)
HEAD_TURN_MARGIN = 0.08      # Allowable head movement from calibration baseline
FLICKER_IGNORE_TIME = 0.1    # Ignore brief flickers below threshold (reduced for faster response)
STATE_LOCK_TIME = 0.15       # Minimum time before state can change again (reduced to prevent getting stuck)
BOTH_BROWS_GRACE_PERIOD = 0.12  # Time after BOTH_BROWS before PREVIOUS can trigger (reduced to 120ms)
PREVIOUS_THRESHOLD_PERCENT = 0.15  # One eyebrow must be 15% higher than the other for PREVIOUS
FAR_DISTANCE_THRESHOLD = 0.35  # Distance ratio threshold for "far" position
SHOCK_SPIKE_THRESHOLD = 0.25  # Maximum allowed eyebrow spike for shock detection (increased)

# New features
AUTO_RECALIBRATE = False     # Auto-recalibration toggle
RECALIBRATION_THRESHOLD = 0.05  # Head offset threshold for auto-recalibration
POSITION_SENSITIVITY_FACTOR = 0.6  # 60% sensitivity in different positions
EYEBROW_VISIBILITY_TIMEOUT = 2.0  # Time before calling out invisible eyebrow
DYNAMIC_NORMALIZATION = True  # Continuously adjust for eyebrow drift
SHOW_GUI = True  # Show/hide GUI text

CAMERA_INDEX = 0
# ------------------------------

pyautogui.FAILSAFE = False

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(CAMERA_INDEX)

# Resizable window
cv2.namedWindow("Shorts Controller", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Shorts Controller", 960, 540)

# State machine states
STATE_LEARNING = "LEARNING"
STATE_NEUTRAL = "NEUTRAL"
STATE_ONE_BROW = "ONE_BROW"
STATE_BOTH_BROWS = "BOTH_BROWS"

current_state = STATE_LEARNING

# Calibration data
left_baseline_samples = []
right_baseline_samples = []
left_baseline = None
right_baseline = None
left_threshold = None
right_threshold = None
unified_baseline = None  # Unified baseline for both eyebrows
left_offset_from_unified = None  # Left eyebrow offset from unified baseline
right_offset_from_unified = None  # Right eyebrow offset from unified baseline
dynamic_normalization_samples = deque(maxlen=30)  # For continuous normalization
manual_threshold_enabled = False  # Manual threshold toggle
manual_threshold_value = None  # Stored manual threshold value

# Head orientation baseline
head_baseline_x = None
head_baseline_y = None

# Gesture tracking
left_brow_history = deque(maxlen=5)
right_brow_history = deque(maxlen=5)

gesture_start_time = None
last_scroll_time = 0

# Flicker debouncing
left_drop_time = None
right_drop_time = None
left_raise_time = None
right_raise_time = None

# State locking
last_state_change_time = 0
last_both_brows_time = -10  # Initialize to negative to allow immediate PREVIOUS on startup
previous_state = STATE_NEUTRAL  # Track previous state for better transition handling

# New feature tracking
face_distance_baseline = None
current_face_distance = None
eyebrow_invisible_start_time = None
left_invisible = False
right_invisible = False
last_warning_time = 0  # For warning message cooldown

# For display
left_smooth = 0
right_smooth = 0
last_action = None
last_action_time = 0
head_facing_camera = True  # Track if user is facing camera


def calculate_thresholds(current_distance=None):
    """Calculate personalized thresholds based on user's baseline."""
    global left_threshold, right_threshold
    if unified_baseline is None:
        return
    
    # Use manual threshold if enabled
    if manual_threshold_enabled and manual_threshold_value is not None:
        left_threshold = manual_threshold_value
        right_threshold = manual_threshold_value
        return
    
    # Use a multiplier to account for natural movement range
    sensitivity = THRESHOLD_MULTIPLIER
    if current_distance and face_distance_baseline:
        distance_ratio = current_distance / face_distance_baseline
        # If face is significantly closer/further, adjust sensitivity
        if abs(distance_ratio - 1.0) > 0.1:  # More than 10% change
            # If far (distance ratio < 0.65, meaning face is smaller/further), use 30% reduction
            if distance_ratio < 0.65:  # Face is far away
                sensitivity = THRESHOLD_MULTIPLIER * 0.7  # 30% lower threshold
            else:
                sensitivity = THRESHOLD_MULTIPLIER * POSITION_SENSITIVITY_FACTOR  # Apply position sensitivity factor
    
    # Both eyebrows use the same unified threshold
    left_threshold = unified_baseline * sensitivity
    right_threshold = unified_baseline * sensitivity


def smooth_measurement(history):
    """Smooth noisy measurements using moving average."""
    if not history:
        return 0
    return sum(history) / len(history)


def can_scroll():
    """Check if cooldown period has passed."""
    return time.time() - last_scroll_time > GESTURE_COOLDOWN


def trigger_scroll(direction):
    """Execute scroll and update cooldown."""
    global last_scroll_time, last_action, last_action_time
    if direction == "NEXT":
        pyautogui.scroll(-SCROLL_AMOUNT)
        print(f"NEXT triggered (scrolling down with {-SCROLL_AMOUNT})")
    elif direction == "PREVIOUS":
        pyautogui.scroll(SCROLL_AMOUNT)
        print(f"PREVIOUS triggered (scrolling up with {SCROLL_AMOUNT})")
    last_scroll_time = time.time()
    last_action = direction
    last_action_time = time.time()


def get_eyebrow_distances(face):
    """Calculate eyebrow-to-eyelid distances for left and right."""
    left_brow = face.landmark[105]
    right_brow = face.landmark[334]
    left_lid = face.landmark[159]
    right_lid = face.landmark[386]
    
    left_distance = abs(left_brow.y - left_lid.y)
    right_distance = abs(right_brow.y - right_lid.y)
    
    return left_distance, right_distance


def are_eyes_closed(face):
    """Check if eyes are closed or blinking by measuring eye openness."""
    # Use upper and lower eyelid landmarks
    left_upper = face.landmark[159]
    left_lower = face.landmark[145]
    right_upper = face.landmark[386]
    right_lower = face.landmark[374]
    
    # Calculate eye openness (vertical distance)
    left_eye_openness = abs(left_upper.y - left_lower.y)
    right_eye_openness = abs(right_upper.y - right_lower.y)
    
    # Eyes are considered closed if openness is very small (lowered to catch blinks)
    eye_closed_threshold = 0.01  # Lowered from 0.02 to catch blinks
    left_closed = left_eye_openness < eye_closed_threshold
    right_closed = right_eye_openness < eye_closed_threshold
    
    return left_closed or right_closed


def get_head_orientation(face):
    """Detect if user is facing camera or looking away using nose position relative to baseline."""
    global head_baseline_x, head_baseline_y
    nose = face.landmark[1]  # Nose tip
    left_ear = face.landmark[234]  # Left ear
    right_ear = face.landmark[454]  # Right ear
    
    # Calculate center point between ears
    ear_center_x = (left_ear.x + right_ear.x) / 2
    ear_center_y = (left_ear.y + right_ear.y) / 2
    
    # Calculate nose position relative to ear center
    nose_x = nose.x - ear_center_x
    nose_y = nose.y - ear_center_y
    
    # If no baseline yet, always return True (we're still learning)
    if head_baseline_x is None or head_baseline_y is None:
        return True, nose_x, nose_y
    
    # Check how far current position is from baseline
    x_offset = abs(nose_x - head_baseline_x)
    y_offset = abs(nose_y - head_baseline_y)
    
    # User is facing camera if within allowable margin from baseline
    facing = x_offset < HEAD_TURN_MARGIN and y_offset < HEAD_TURN_MARGIN
    
    return facing, x_offset, y_offset


def get_face_distance(face, frame_width, frame_height):
    """Calculate face distance from camera using actual pixel measurements."""
    # Get face landmarks
    landmarks = face.landmark
    
    # Calculate bounding box in actual pixels
    min_x = min(lm.x for lm in landmarks) * frame_width
    max_x = max(lm.x for lm in landmarks) * frame_width
    min_y = min(lm.y for lm in landmarks) * frame_height
    max_y = max(lm.y for lm in landmarks) * frame_height
    
    # Face size as distance indicator (larger = closer)
    face_width = max_x - min_x
    face_height = max_y - min_y
    face_size = (face_width + face_height) / 2
    
    return face_size


def check_eyebrow_visibility(face):
    """Check if eyebrows are visible in the frame with more lenient bounds."""
    left_brow = face.landmark[105]
    right_brow = face.landmark[334]
    
    # More lenient bounds to handle different head positions (looking down at videos)
    # Expanded y-range from 0.05-0.95 to 0.0-1.0 to handle looking down
    left_visible = (0.02 < left_brow.x < 0.98 and 0.0 < left_brow.y < 1.0)
    right_visible = (0.02 < right_brow.x < 0.98 and 0.0 < right_brow.y < 1.0)
    
    return left_visible, right_visible


def should_auto_recalibrate(head_offset_x, head_offset_y):
    """Check if head position changed enough to trigger auto-recalibration based on head offset."""
    global head_baseline_x, head_baseline_y
    if head_baseline_x is None or head_baseline_y is None:
        return False
    
    # Calculate total head offset from baseline
    total_offset = (abs(head_offset_x) + abs(head_offset_y)) / 2
    
    return total_offset > RECALIBRATION_THRESHOLD


def reset_calibration():
    """Reset calibration to learning state."""
    global current_state, left_baseline, right_baseline, left_threshold, right_threshold, head_baseline_x, head_baseline_y, last_state_change_time, last_both_brows_time, last_warning_time, unified_baseline, left_offset_from_unified, right_offset_from_unified, manual_threshold_enabled, manual_threshold_value
    current_state = STATE_LEARNING
    left_baseline_samples.clear()
    right_baseline_samples.clear()
    left_baseline = None
    right_baseline = None
    left_threshold = None
    right_threshold = None
    head_baseline_x = None
    head_baseline_y = None
    left_brow_history.clear()
    right_brow_history.clear()
    dynamic_normalization_samples.clear()
    last_state_change_time = 0
    last_both_brows_time = -10  # Initialize to negative to allow immediate PREVIOUS on startup
    last_warning_time = 0
    unified_baseline = None
    left_offset_from_unified = None
    right_offset_from_unified = None
    # Keep manual threshold enabled across recalibration, just reset value
    manual_threshold_value = None
    print("Recalibrating...")


def manual_threshold_calibration():
    """Toggle manual threshold calibration using current eyebrow level."""
    global manual_threshold_enabled, manual_threshold_value, left_threshold, right_threshold, unified_baseline
    if unified_baseline is None:
        print("Must complete calibration first!")
        return
    
    if manual_threshold_enabled:
        # Disable manual threshold
        manual_threshold_enabled = False
        manual_threshold_value = None
        print("Manual threshold DISABLED - using automatic thresholds")
        # Recalculate automatic thresholds
        calculate_thresholds()
    else:
        # Enable manual threshold with current level
        current_left = left_smooth
        current_right = right_smooth
        current_avg = (current_left + current_right) / 2
        
        manual_threshold_value = current_avg
        manual_threshold_enabled = True
        left_threshold = current_avg
        right_threshold = current_avg
        
        print(f"Manual threshold ENABLED - set to: {current_avg:.4f}")
        print(f"Previous unified baseline: {unified_baseline:.4f}")
        print(f"Current eyebrow levels - Left: {current_left:.4f}, Right: {current_right:.4f}")


print("Controls:")
print("C = Recalibrate")
print("' = Toggle Auto-Recalibration")
print("\\ = Toggle Manual Threshold")
print(", = Toggle GUI")
print("Q = Quit")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    # ================= FACE PROCESSING =================
    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]

        left_dist, right_dist = get_eyebrow_distances(face)
        head_facing_camera, head_offset_x, head_offset_y = get_head_orientation(face)
        current_face_distance = get_face_distance(face, w, h)
        left_visible, right_visible = check_eyebrow_visibility(face)
        eyes_closed = are_eyes_closed(face)
        
        # Normalize eyebrow distances to unified baseline
        if unified_baseline is not None and left_offset_from_unified is not None and right_offset_from_unified is not None:
            # Remove the offsets so both eyebrows start from the same baseline
            left_dist = left_dist - left_offset_from_unified
            right_dist = right_dist - right_offset_from_unified
            
            # Dynamic normalization: continuously adjust for eyebrow drift and bridge gap
            if DYNAMIC_NORMALIZATION and current_state == STATE_NEUTRAL:
                dynamic_normalization_samples.append((left_dist, right_dist))
                if len(dynamic_normalization_samples) >= 20:
                    # Calculate current average offsets
                    avg_left = sum(s[0] for s in dynamic_normalization_samples) / len(dynamic_normalization_samples)
                    avg_right = sum(s[1] for s in dynamic_normalization_samples) / len(dynamic_normalization_samples)
                    
                    # Calculate drift from unified baseline
                    current_avg = (avg_left + avg_right) / 2
                    drift = current_avg - unified_baseline
                    
                    # Calculate gap between eyebrows
                    eyebrow_gap = abs(avg_left - avg_right)
                    
                    # Adjust offsets if significant drift detected
                    if abs(drift) > 0.01:  # Only adjust if drift is significant
                        # Gradually adjust offsets to compensate for drift
                        adjustment_factor = 0.1  # Slow adjustment
                        left_offset_from_unified += drift * adjustment_factor
                        right_offset_from_unified += drift * adjustment_factor
                    
                    # Micro-adjustments to bridge the gap between eyebrows
                    if eyebrow_gap > 0.03:  # If gap is significant (increased from 0.02)
                        # Calculate how much to adjust each side
                        gap_adjustment = eyebrow_gap * 0.4  # Adjust 40% of the gap (increased from 30%)
                        micro_factor = 0.1  # Faster adjustment to handle head tilt better (increased from 0.05)
                        
                        # Bring the higher one down and the lower one up
                        if avg_left > avg_right:
                            left_offset_from_unified += gap_adjustment * micro_factor
                            right_offset_from_unified -= gap_adjustment * micro_factor
                        else:
                            left_offset_from_unified -= gap_adjustment * micro_factor
                            right_offset_from_unified += gap_adjustment * micro_factor

        # Smooth measurements
        left_brow_history.append(left_dist)
        right_brow_history.append(right_dist)

        left_smooth = smooth_measurement(left_brow_history)
        right_smooth = smooth_measurement(right_brow_history)

        # ================= LEARNING STATE =================
        if current_state == STATE_LEARNING:
            left_baseline_samples.append(left_smooth)
            right_baseline_samples.append(right_smooth)

            # Also collect head orientation baseline
            nose = face.landmark[1]
            left_ear = face.landmark[234]
            right_ear = face.landmark[454]
            ear_center_x = (left_ear.x + right_ear.x) / 2
            ear_center_y = (left_ear.y + right_ear.y) / 2
            current_head_x = nose.x - ear_center_x
            current_head_y = nose.y - ear_center_y
            
            # Collect face distance baseline
            current_face_distance = get_face_distance(face, w, h)

            # Check if we have enough samples
            if len(left_baseline_samples) >= CALIBRATION_FRAMES:
                left_baseline = smooth_measurement(left_baseline_samples)
                right_baseline = smooth_measurement(right_baseline_samples)
                
                # Calculate unified baseline (average of both eyebrows)
                unified_baseline = (left_baseline + right_baseline) / 2
                
                # Calculate offsets from unified baseline
                left_offset_from_unified = left_baseline - unified_baseline
                right_offset_from_unified = right_baseline - unified_baseline
                
                calculate_thresholds(current_face_distance)
                
                # Set head baseline to final measurement
                head_baseline_x = current_head_x
                head_baseline_y = current_head_y
                
                # Set face distance baseline
                face_distance_baseline = current_face_distance
                
                current_state = STATE_NEUTRAL
                print(f"Personalized! Left baseline: {left_baseline:.4f}, Right baseline: {right_baseline:.4f}")
                print(f"Unified baseline: {unified_baseline:.4f}")
                print(f"Offsets - Left: {left_offset_from_unified:.4f}, Right: {right_offset_from_unified:.4f}")
                print(f"Head baseline set: x={head_baseline_x:.4f}, y={head_baseline_y:.4f}")
                print(f"Face distance baseline: {face_distance_baseline:.4f}")

        # ================= GESTURE DETECTION =================
        elif current_state != STATE_LEARNING:
            # Dynamic threshold adjustment based on face distance
            if current_face_distance and face_distance_baseline:
                calculate_thresholds(current_face_distance)
            
            # Check for auto-recalibration if enabled (based on head offset)
            if AUTO_RECALIBRATE and should_auto_recalibrate(head_offset_x, head_offset_y):
                print("Auto-recalibrating due to head position change...")
                reset_calibration()
                continue
            
            # Check eyebrow visibility
            if not left_visible or not right_visible:
                if eyebrow_invisible_start_time is None:
                    eyebrow_invisible_start_time = time.time()
                    left_invisible = not left_visible
                    right_invisible = not right_visible
                elif time.time() - eyebrow_invisible_start_time > EYEBROW_VISIBILITY_TIMEOUT:
                    # Call out missing eyebrow after 2 seconds, but only once every 5 seconds
                    if time.time() - last_warning_time > 5.0:
                        missing = []
                        if left_invisible:
                            missing.append("LEFT")
                        if right_invisible:
                            missing.append("RIGHT")
                        print(f"WARNING: {', '.join(missing)} EYEBROW NOT VISIBLE")
                        last_warning_time = time.time()
                    # Reset to neutral and stop gesture detection
                    current_state = STATE_NEUTRAL
                    gesture_start_time = None
            else:
                # Reset visibility timer when both eyebrows are visible
                eyebrow_invisible_start_time = None
                left_invisible = False
                right_invisible = False
            
            # Check if eyes are closed - prevent gesture detection
            if eyes_closed:
                current_state = STATE_NEUTRAL
                gesture_start_time = None
                # Optionally add visual feedback for closed eyes

            
            # Only detect gestures if user is facing camera and both eyebrows visible
            if not head_facing_camera or not left_visible or not right_visible:
                # Reset to neutral if looking away or eyebrow not visible
                current_state = STATE_NEUTRAL
                gesture_start_time = None
            else:
                # Check if eyebrows are raised above personalized thresholds with hysteresis
                # Use percentage-based margin to adapt to measurement scale
                left_raised_threshold = left_threshold * (1 + HYSTERESIS_PERCENT)
                right_raised_threshold = right_threshold * (1 + HYSTERESIS_PERCENT)
                # More lenient "below" threshold - just needs to be under base threshold
                left_below_threshold = left_threshold * 0.95  # Only 5% below base threshold
                right_below_threshold = right_threshold * 0.95
                
                left_raised = left_smooth > left_raised_threshold
                right_raised = right_smooth > right_raised_threshold
                
                # Also check if clearly below threshold for return to neutral
                left_below = left_smooth < left_below_threshold
                right_below = right_smooth < right_below_threshold

                # Debouncing: track when eyebrows drop below threshold
                # Only track if eyebrow is visible
                if not left_raised and left_visible:
                    if left_drop_time is None:
                        left_drop_time = time.time()
                else:
                    left_drop_time = None  # Reset if eyebrow goes back up
                
                if not right_raised and right_visible:
                    if right_drop_time is None:
                        right_drop_time = time.time()
                else:
                    right_drop_time = None  # Reset if eyebrow goes back up

                # Debouncing: track when eyebrows raise above threshold
                # Only track if eyebrow is visible
                if left_raised and left_visible:
                    if left_raise_time is None:
                        left_raise_time = time.time()
                else:
                    left_raise_time = None  # Reset if eyebrow drops back down or not visible
                
                if right_raised and right_visible:
                    if right_raise_time is None:
                        right_raise_time = time.time()
                else:
                    right_raise_time = None  # Reset if eyebrow drops back down or not visible

                # Determine if eyebrows are truly raised/dropped (stayed consistent long enough)
                # Only consider raised if visible
                left_truly_raised = left_raised and left_visible and (left_raise_time and (time.time() - left_raise_time) > FLICKER_IGNORE_TIME)
                right_truly_raised = right_raised and right_visible and (right_raise_time and (time.time() - right_raise_time) > FLICKER_IGNORE_TIME)
                left_truly_dropped = left_below and left_visible and (left_drop_time and (time.time() - left_drop_time) > FLICKER_IGNORE_TIME)
                right_truly_dropped = right_below and right_visible and (right_drop_time and (time.time() - right_drop_time) > FLICKER_IGNORE_TIME)
                
                # Shock spike detection: ignore sudden large spikes in eyebrow movement
                # Calculate spike magnitude
                left_spike = abs(left_smooth - left_threshold) if left_threshold else 0
                right_spike = abs(right_smooth - right_threshold) if right_threshold else 0
                max_spike = max(left_spike, right_spike)
                
                # If spike is too large, treat as shock/reaction and ignore
                if max_spike > SHOCK_SPIKE_THRESHOLD:
                    # Reset state to neutral and ignore this gesture
                    current_state = STATE_NEUTRAL
                    gesture_start_time = None
                    left_raise_time = None
                    right_raise_time = None
                    left_drop_time = None
                    right_drop_time = None
                    continue

                # State machine with locking to prevent flickering
                time_since_state_change = time.time() - last_state_change_time
                
                if current_state == STATE_NEUTRAL:
                    if left_truly_raised and right_truly_raised:
                        current_state = STATE_BOTH_BROWS
                        gesture_start_time = time.time()
                        last_state_change_time = time.time()
                        last_both_brows_time = time.time()
                        previous_state = STATE_NEUTRAL
                        left_drop_time = None
                        right_drop_time = None
                        left_raise_time = None
                        right_raise_time = None
                    elif left_truly_raised or right_truly_raised:
                        # Only allow ONE_BROW if enough time passed since BOTH_BROWS
                        if time.time() - last_both_brows_time > BOTH_BROWS_GRACE_PERIOD:
                            current_state = STATE_ONE_BROW
                            gesture_start_time = time.time()
                            last_state_change_time = time.time()
                            previous_state = STATE_NEUTRAL
                            left_drop_time = None
                            right_drop_time = None
                            left_raise_time = None
                            right_raise_time = None

                elif current_state == STATE_ONE_BROW:
                    if left_truly_raised and right_truly_raised:
                        current_state = STATE_BOTH_BROWS
                        gesture_start_time = time.time()
                        last_state_change_time = time.time()
                        last_both_brows_time = time.time()
                        previous_state = STATE_ONE_BROW
                        left_drop_time = None
                        right_drop_time = None
                        left_raise_time = None
                        right_raise_time = None
                    elif left_truly_dropped and right_truly_dropped:
                        # Clearly returned to neutral - trigger PREVIOUS if held briefly but not too long
                        gesture_duration = time.time() - gesture_start_time if gesture_start_time else 0
                        time_since_both_brows = time.time() - last_both_brows_time
                        
                        # Simple PREVIOUS trigger with grace period
                        if (gesture_duration > GESTURE_HOLD_TIME and 
                            gesture_duration < GESTURE_MAX_DURATION and
                            time_since_both_brows > BOTH_BROWS_GRACE_PERIOD and
                            can_scroll()):
                            trigger_scroll("PREVIOUS")
                        
                        current_state = STATE_NEUTRAL
                        last_state_change_time = time.time()
                        previous_state = STATE_ONE_BROW
                        left_drop_time = None
                        right_drop_time = None
                        left_raise_time = None
                        right_raise_time = None
                    # Fixed: If the raised eyebrow drops (but other stays neutral), trigger PREVIOUS
                    elif (left_truly_dropped and not right_raised) or (right_truly_dropped and not left_raised):
                        gesture_duration = time.time() - gesture_start_time if gesture_start_time else 0
                        time_since_both_brows = time.time() - last_both_brows_time
                        
                        # Simple trigger with max duration check
                        if (gesture_duration > GESTURE_HOLD_TIME and 
                            gesture_duration < GESTURE_MAX_DURATION and
                            can_scroll()):
                            trigger_scroll("PREVIOUS")
                        
                        current_state = STATE_NEUTRAL
                        last_state_change_time = time.time()
                        previous_state = STATE_ONE_BROW
                        left_drop_time = None
                        right_drop_time = None
                        left_raise_time = None
                        right_raise_time = None
                    # Timeout: reset to neutral if stuck too long
                    elif gesture_start_time and (time.time() - gesture_start_time) > GESTURE_MAX_DURATION:
                        current_state = STATE_NEUTRAL
                        last_state_change_time = time.time()
                        previous_state = STATE_ONE_BROW
                        left_drop_time = None
                        right_drop_time = None
                        left_raise_time = None
                        right_raise_time = None

                elif current_state == STATE_BOTH_BROWS:
                    if left_truly_dropped and right_truly_dropped:
                        # Clearly returned to neutral - trigger NEXT if held briefly but not too long
                        gesture_duration = time.time() - gesture_start_time if gesture_start_time else 0
                        if (gesture_duration > GESTURE_HOLD_TIME and 
                            gesture_duration < GESTURE_MAX_DURATION and
                            can_scroll()):
                            trigger_scroll("NEXT")
                        current_state = STATE_NEUTRAL
                        last_state_change_time = time.time()
                        previous_state = STATE_BOTH_BROWS
                        left_drop_time = None
                        right_drop_time = None
                        left_raise_time = None
                        right_raise_time = None
                    # Don't transition to ONE_BROW - stay in BOTH_BROWS until both drop
                    # This prevents triggering PREVIOUS when dropping from BOTH_BROWS
                    else:
                        # If stuck in BOTH_BROWS too long without dropping, reset to neutral
                        if gesture_start_time and (time.time() - gesture_start_time) > GESTURE_MAX_DURATION:
                            current_state = STATE_NEUTRAL
                            last_state_change_time = time.time()
                            previous_state = STATE_BOTH_BROWS
                            left_drop_time = None
                            right_drop_time = None
                            left_raise_time = None
                            right_raise_time = None

    # ================= UI DISPLAY =================
    if SHOW_GUI:
        # Status (top left)
        if current_state == STATE_LEARNING:
            status_text = f"LEARNING... {len(left_baseline_samples)}/{CALIBRATION_FRAMES}"
            status_color = (0, 100, 255)
        else:
            status_text = "PERSONALIZED"
            status_color = (0, 200, 100)

        cv2.putText(frame, status_text, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 2)

        # Eyebrow measurements (left side)
        if current_state != STATE_LEARNING:
            left_raised_threshold = left_threshold * (1 + HYSTERESIS_PERCENT)
            right_raised_threshold = right_threshold * (1 + HYSTERESIS_PERCENT)
            
            # Left column
            cv2.putText(frame, f"L: {left_smooth:.4f}", (15, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"R: {right_smooth:.4f}", (15, 95), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
            # Trigger thresholds
            cv2.putText(frame, f"L thr: {left_raised_threshold:.4f}", (15, 120), cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, f"R thr: {right_raised_threshold:.4f}", (15, 145), cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 200), 1)

            # Raised status
            left_display_raised = left_smooth > left_raised_threshold
            right_display_raised = right_smooth > right_raised_threshold
            left_color = (0, 255, 0) if left_display_raised else (80, 80, 80)
            right_color = (0, 255, 0) if right_display_raised else (80, 80, 80)
            cv2.putText(frame, f"L: {'UP' if left_display_raised else 'DN'}", (15, 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, left_color, 2)
            cv2.putText(frame, f"R: {'UP' if right_display_raised else 'DN'}", (15, 195), cv2.FONT_HERSHEY_DUPLEX, 0.5, right_color, 2)

            # Current gesture state
            gesture_text = current_state
            if current_state == STATE_NEUTRAL:
                gesture_color = (200, 200, 200)
            elif current_state == STATE_ONE_BROW:
                gesture_color = (255, 255, 0)
            elif current_state == STATE_BOTH_BROWS:
                gesture_color = (0, 255, 255)
            cv2.putText(frame, gesture_text, (15, 230), cv2.FONT_HERSHEY_DUPLEX, 0.7, gesture_color, 2)

            # Last action
            if last_action and (time.time() - last_action_time) < 2.0:
                action_color = (0, 255, 0) if last_action == "NEXT" else (255, 255, 0)
                cv2.putText(frame, last_action, (15, 260), cv2.FONT_HERSHEY_DUPLEX, 0.9, action_color, 3)

        # Right column info
        if current_state != STATE_LEARNING:
            # Head orientation
            head_color = (0, 255, 0) if head_facing_camera else (0, 100, 255)
            head_status = "FACE" if head_facing_camera else "AWAY"
            cv2.putText(frame, f"HEAD: {head_status}", (w - 120, 35), cv2.FONT_HERSHEY_DUPLEX, 0.5, head_color, 1)
            
            # Head offset
            if head_baseline_x is not None and head_baseline_y is not None:
                cv2.putText(frame, f"OFF: {head_offset_x:.3f}", (w - 120, 60), cv2.FONT_HERSHEY_DUPLEX, 0.45, (180, 180, 180), 1)
            
            # Face distance
            if face_distance_baseline is not None and current_face_distance is not None:
                distance_ratio = current_face_distance / face_distance_baseline
                dist_color = (100, 255, 100) if 0.8 < distance_ratio < 1.2 else (255, 200, 100)
                cv2.putText(frame, f"DIST: {distance_ratio:.2f}x", (w - 120, 85), cv2.FONT_HERSHEY_DUPLEX, 0.5, dist_color, 1)
            
            # Eyebrow visibility
            vis_color = (0, 255, 0) if (left_visible and right_visible) else (0, 100, 255)
            vis_text = "VIS" if (left_visible and right_visible) else "HID"
            cv2.putText(frame, f"BROW: {vis_text}", (w - 120, 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, vis_color, 1)
            
            # Auto-recalibration
            auto_color = (0, 255, 0) if AUTO_RECALIBRATE else (100, 100, 100)
            cv2.putText(frame, f"AUTO: {'ON' if AUTO_RECALIBRATE else 'OFF'}", (w - 120, 135), cv2.FONT_HERSHEY_DUPLEX, 0.45, auto_color, 1)
            
            # Dynamic normalization
            dynamic_color = (0, 255, 0) if DYNAMIC_NORMALIZATION else (100, 100, 100)
            cv2.putText(frame, f"DYN: {'ON' if DYNAMIC_NORMALIZATION else 'OFF'}", (w - 120, 160), cv2.FONT_HERSHEY_DUPLEX, 0.45, dynamic_color, 1)
            
            # Manual threshold
            manual_color = (0, 255, 0) if manual_threshold_enabled else (100, 100, 100)
            manual_text = f"MAN: {'ON' if manual_threshold_enabled else 'OFF'}"
            if manual_threshold_enabled and manual_threshold_value is not None:
                manual_text += f" {manual_threshold_value:.3f}"
            cv2.putText(frame, manual_text, (w - 120, 185), cv2.FONT_HERSHEY_DUPLEX, 0.45, manual_color, 1)
            
            # Offsets
            if unified_baseline is not None and left_offset_from_unified is not None and right_offset_from_unified is not None:
                cv2.putText(frame, f"OFF L:{left_offset_from_unified:.3f}", (w - 120, 210), cv2.FONT_HERSHEY_DUPLEX, 0.4, (180, 180, 180), 1)
                cv2.putText(frame, f"OFF R:{right_offset_from_unified:.3f}", (w - 120, 235), cv2.FONT_HERSHEY_DUPLEX, 0.4, (180, 180, 180), 1)
        
        # Controls hint (bottom)
        cv2.putText(frame, "C=Recal ' =Auto ,=Hide", (15, h - 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (150, 150, 150), 1)
        pass  # Handle any GUI errors gracefully

    cv2.imshow("Shorts Controller", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        reset_calibration()

    elif key == ord("'"):
        AUTO_RECALIBRATE = not AUTO_RECALIBRATE
        print(f"Auto-recalibration: {'ON' if AUTO_RECALIBRATE else 'OFF'}")

    elif key == ord("\\"):  # Backslash key
        manual_threshold_calibration()

    elif key == ord(","):  # Comma key
        SHOW_GUI = not SHOW_GUI
        print(f"GUI: {'VISIBLE' if SHOW_GUI else 'HIDDEN'}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
