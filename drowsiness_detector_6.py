import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from scipy.spatial import distance
from collections import deque
import streamlit as st

class PersonalizedDrowsinessDetector:
    """
    Adaptive drowsiness detection system that personalizes to individual users.
    Eliminates bias by establishing user-specific baselines rather than absolute thresholds.
    Now includes head droop detection in addition to eye-closure and head-tilt detection.
    """
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Eye landmark indices for MediaPipe Face Mesh
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Head pose landmarks — standard symmetric 6-point set for MediaPipe solvePnP
        # (nose tip, chin, left eye outer, right eye outer, left mouth, right mouth)
        self.HEAD_POSE_LANDMARKS = [1, 152, 234, 454, 58, 284]
        
        # User baseline data
        self.user_baseline = None
        self.calibration_data = {
            'ear_values': [],
            'blink_durations': [],
            'head_poses': []          # stores (x_rot, y_rot) tuples
        }
        
        # Real-time tracking
        self.ear_history = deque(maxlen=30)   # Last ~1 second at 30 fps
        self.pitch_history = deque(maxlen=30) # Last ~1 second of pitch angles
        self.blink_counter = 0
        self.blink_start_time = None
        self.total_blinks = 0
        self.drowsiness_score = 0
        
        # Drowsiness thresholds (will be personalized)
        self.ear_threshold = None
        self.blink_duration_threshold = None
        self.consecutive_frames_threshold = 20   # ~0.67 s for eye closure
        self.consecutive_low_ear = 0

        # ── Head droop thresholds (will be personalized during calibration) ──
        self.baseline_pitch = None          # user's normal pitch angle
        self.pitch_droop_threshold = None   # pitch value below which droop is flagged
        self.droop_frames_threshold = 15    # ~0.5 s of sustained droop before alert
        self.consecutive_droop_frames = 0   # running counter

        # ── Sideways tilt thresholds (personalised during calibration) ──
        self.baseline_yaw = None            # user's neutral yaw (rarely exactly 0)
        self.yaw_left_threshold  = None     # baseline_yaw - offset  (left turn)
        self.yaw_right_threshold = None     # baseline_yaw + offset  (right turn)
        self.tilt_frames_threshold = 15     # ~0.5 s of sustained tilt before alert
        self.consecutive_tilt_frames = 0    # running counter

    # ─────────────────────────────────────────────
    # Core computation helpers
    # ─────────────────────────────────────────────

    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for given eye landmarks."""
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_head_pose(self, landmarks, image_shape):
        """
        Estimate head pitch and yaw purely from landmark geometry.
        Avoids solvePnP which produces numerically unstable results with
        MediaPipe's normalized z-depth values.

        Pitch: vertical offset of nose tip relative to the eye-chin midpoint.
               Negative = chin dropping down (droop).
        Yaw:   horizontal offset of nose tip relative to face centre.
               Negative = turned left, Positive = turned right.

        Both values are expressed as a ratio of face height, then scaled to
        produce degree-like units that are stable and symmetric.
        """
        h, w = image_shape[:2]

        def pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        nose      = pt(1)    # nose tip
        chin      = pt(152)  # chin
        left_eye  = pt(33)   # right eye outer corner (mirrored frame)
        right_eye = pt(263)  # left  eye outer corner (mirrored frame)
        forehead  = pt(10)   # top of forehead

        # Face geometry references
        eye_mid      = (left_eye + right_eye) / 2.0
        face_height  = np.linalg.norm(chin - forehead) + 1e-6
        face_width   = np.linalg.norm(right_eye - left_eye) + 1e-6

        # Pitch: how far the nose sits below the eye midpoint, normalised by
        # face height and scaled so that ~8-10 units ≈ a noticeable droop.
        vertical_offset = nose[1] - eye_mid[1]          # positive = nose below eyes
        pitch = -(vertical_offset / face_height) * 100  # negate so droop → negative

        # Yaw: horizontal offset of nose from the eye midpoint, normalised by
        # face width. Negative = nose shifted left = head turned left.
        horizontal_offset = nose[0] - eye_mid[0]
        yaw = (horizontal_offset / face_width) * 100

        return pitch, yaw

    # ─────────────────────────────────────────────
    # Calibration
    # ─────────────────────────────────────────────

    def calibrate(self, duration=10):
        """
        Calibrate the system for the current user.
        User should maintain normal, alert driving posture during calibration.
        """
        st.info(f"🎯 Calibration Phase: Please look at the camera naturally for {duration} seconds.")
        st.info("Keep your eyes open normally and maintain a comfortable, upright posture.")
        
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        frame_count = 0
        
        calibration_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # EAR
                left_eye  = [(landmarks[i].x, landmarks[i].y) for i in self.LEFT_EYE]
                right_eye = [(landmarks[i].x, landmarks[i].y) for i in self.RIGHT_EYE]
                left_ear  = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                avg_ear   = (left_ear + right_ear) / 2.0
                self.calibration_data['ear_values'].append(avg_ear)
                
                # Head pose
                pitch, yaw = self.get_head_pose(landmarks, frame.shape)
                self.calibration_data['head_poses'].append((pitch, yaw))
                
                cv2.putText(frame,
                            f"Calibrating... {int(duration - (time.time() - start_time))}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame,
                            f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                frame_count += 1
            
            progress_bar.progress(min((time.time() - start_time) / duration, 1.0))
            calibration_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        cap.release()
        progress_bar.empty()
        calibration_placeholder.empty()
        
        if len(self.calibration_data['ear_values']) > 0:
            # ── EAR threshold ──
            ear_values = np.array(self.calibration_data['ear_values'])
            self.ear_threshold = np.mean(ear_values) - 1.5 * np.std(ear_values)
            self.ear_threshold = max(0.15, min(0.25, self.ear_threshold))

            # ── Head droop threshold ──
            # Collect all pitch values recorded during calibration
            pitch_values = np.array([p for p, _ in self.calibration_data['head_poses']])
            self.baseline_pitch = float(np.mean(pitch_values))
            pitch_std = float(np.std(pitch_values))

            # A droop is flagged when pitch falls more than 2.5 std below the
            # user's own normal pitch (i.e. chin drops noticeably).
            # Enforce a minimum drop of 8 units so minor wobbles don't trigger it.
            droop_drop = max(8.0, 2.5 * pitch_std)
            self.pitch_droop_threshold = self.baseline_pitch - droop_drop
            
            # ── Yaw (sideways tilt) thresholds ──
            # Calibrate baseline yaw so asymmetric RQDecomp3x3 output is corrected
            # per-user before applying the ±20° offset.
            yaw_values = np.array([y for _, y in self.calibration_data['head_poses']])
            self.baseline_yaw = float(np.mean(yaw_values))
            yaw_offset = 20.0   # degrees either side of the user's own neutral
            self.yaw_left_threshold  = self.baseline_yaw - yaw_offset
            self.yaw_right_threshold = self.baseline_yaw + yaw_offset

            self.user_baseline = {
                'mean_ear':            float(np.mean(ear_values)),
                'std_ear':             float(np.std(ear_values)),
                'ear_threshold':       self.ear_threshold,
                'baseline_pitch':      self.baseline_pitch,
                'pitch_std':           pitch_std,
                'pitch_droop_threshold': self.pitch_droop_threshold,
                'baseline_yaw':        self.baseline_yaw,
                'yaw_left_threshold':  self.yaw_left_threshold,
                'yaw_right_threshold': self.yaw_right_threshold,
                'calibration_frames':  frame_count
            }
            
            st.success(f"✅ Calibration Complete!")
            st.info(f"EAR threshold: {self.ear_threshold:.3f}  |  "
                    f"Baseline pitch: {self.baseline_pitch:.1f}  |  "
                    f"Droop trigger below: {self.pitch_droop_threshold:.1f}  |  "
                    f"Baseline yaw: {self.baseline_yaw:.1f}")
            return True
        else:
            st.error("❌ Calibration failed. Please ensure your face is visible to the camera.")
            return False

    # ─────────────────────────────────────────────
    # Real-time detection
    # ─────────────────────────────────────────────

    def detect_drowsiness(self):
        """Real-time drowsiness detection using personalized thresholds."""
        if self.user_baseline is None:
            st.error("Please complete calibration first!")
            return
        
        st.info("🚗 Drowsiness Detection Active. Press 'Stop' button to end.")
        
        cap = cv2.VideoCapture(0)
        
        video_placeholder   = st.empty()
        status_placeholder  = st.empty()
        metrics_placeholder = st.empty()
        stop_button         = st.button("Stop Detection")
        
        alert_active     = False
        alert_start_time = None
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame     = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = self.face_mesh.process(rgb_frame)
            
            drowsy      = False
            status_text  = "ALERT ✓"
            status_color = (0, 255, 0)
            h, w = frame.shape[:2]
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # ── Eye aspect ratio ──
                left_eye  = [(landmarks[i].x, landmarks[i].y) for i in self.LEFT_EYE]
                right_eye = [(landmarks[i].x, landmarks[i].y) for i in self.RIGHT_EYE]
                left_ear  = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                avg_ear   = (left_ear + right_ear) / 2.0
                self.ear_history.append(avg_ear)
                
                # ── Eye-closure drowsiness ──
                if avg_ear < self.ear_threshold:
                    self.consecutive_low_ear += 1
                    if self.consecutive_low_ear >= self.consecutive_frames_threshold:
                        drowsy       = True
                        status_text  = "⚠️ DROWSINESS DETECTED (Eyes) ⚠️"
                        status_color = (0, 0, 255)
                        if not alert_active:
                            alert_active     = True
                            alert_start_time = time.time()
                else:
                    self.consecutive_low_ear = 0
                    if not drowsy:
                        alert_active = False
                
                # ── Head pose ──
                pitch, yaw = self.get_head_pose(landmarks, frame.shape)
                self.pitch_history.append(pitch)

                # ── Head DROOP detection ──
                # Droop = pitch drops well below the user's calibrated baseline
                if pitch < self.pitch_droop_threshold:
                    self.consecutive_droop_frames += 1
                    if self.consecutive_droop_frames >= self.droop_frames_threshold:
                        drowsy       = True
                        status_text  = "⚠️ HEAD DROOP DETECTED ⚠️"
                        status_color = (0, 100, 255)   # orange-red
                        if not alert_active:
                            alert_active     = True
                            alert_start_time = time.time()
                else:
                    self.consecutive_droop_frames = 0
                
                # ── Excessive sideways tilt / yaw ──
                # Use personalised asymmetric thresholds so left and right are
                # equally sensitive despite RQDecomp3x3 output asymmetry.
                if yaw < self.yaw_left_threshold or yaw > self.yaw_right_threshold:
                    self.consecutive_tilt_frames += 1
                    if self.consecutive_tilt_frames >= self.tilt_frames_threshold:
                        direction    = "LEFT" if yaw < self.yaw_left_threshold else "RIGHT"
                        drowsy       = True
                        status_text  = f"⚠️ HEAD TILT {direction} DETECTED ⚠️"
                        status_color = (0, 165, 255)
                else:
                    self.consecutive_tilt_frames = 0
                
                # ── Draw eye outlines ──
                for eye in [left_eye, right_eye]:
                    eye_pts = np.array([(int(x * w), int(y * h)) for x, y in eye], np.int32)
                    cv2.polylines(frame, [eye_pts], True, (0, 255, 0), 1)

                # ── HUD overlays ──
                cv2.putText(frame, f"EAR: {avg_ear:.2f}  (thr {self.ear_threshold:.2f})",
                            (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Pitch: {pitch:.1f} (base {self.baseline_pitch:.1f} thr {self.pitch_droop_threshold:.1f})",
                            (10, 58),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}",
                            (10, 86),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.putText(frame, status_text,
                            (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

                # ── Directional arrows ──
                nose_lm = landmarks[1]
                nx, ny = int(nose_lm.x * w), int(nose_lm.y * h)

                # Downward arrow on nose tip for head droop
                if self.consecutive_droop_frames >= self.droop_frames_threshold:
                    cv2.arrowedLine(frame, (nx, ny - 20), (nx, ny + 40),
                                    (0, 100, 255), 3, tipLength=0.4)

                # Left / right arrows above forehead for sideways tilt
                if self.consecutive_tilt_frames >= self.tilt_frames_threshold:
                    forehead_lm = landmarks[10]
                    fx = int(forehead_lm.x * w)
                    fy = int(forehead_lm.y * h) - 20   # a little above forehead
                    if yaw < self.yaw_left_threshold:   # turned left → arrow points right (mirrored frame)
                        cv2.arrowedLine(frame, (fx - 50, fy), (fx + 30, fy),
                                        (0, 165, 255), 3, tipLength=0.4)
                    else:                               # turned right → arrow points left (mirrored frame)
                        cv2.arrowedLine(frame, (fx + 50, fy), (fx - 30, fy),
                                        (0, 165, 255), 3, tipLength=0.4)

                # Red border when drowsy
                if drowsy:
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            
            # ── Stream frame ──
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            if drowsy:
                status_placeholder.error(f"🚨 {status_text}")
            else:
                status_placeholder.success(f"✅ {status_text}")
            
            # ── Metrics panel ──
            if len(self.ear_history) > 0 and len(self.pitch_history) > 0:
                col_a, col_b = metrics_placeholder.columns(2)
                col_a.metric(
                    "Current EAR",
                    f"{np.mean(list(self.ear_history)):.3f}",
                    delta=f"{np.mean(list(self.ear_history)) - self.user_baseline['mean_ear']:.3f}"
                )
                col_b.metric(
                    "Head Pitch",
                    f"{np.mean(list(self.pitch_history)):.1f}°",
                    delta=f"{np.mean(list(self.pitch_history)) - self.baseline_pitch:.1f}°"
                )
        
        cap.release()
        video_placeholder.empty()
        status_placeholder.empty()
        metrics_placeholder.empty()
        st.success("Detection stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit app entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Personalized Drowsiness Detector", layout="wide")
    
    st.title("🚗 Personalized Drowsiness Detection System")
    st.markdown("""
    ### An Unbiased, Adaptive Drowsiness Detection System
    
    This system eliminates bias by **personalizing to your unique facial features** rather than 
    using absolute thresholds that may not work for everyone.
    
    **How it works:**
    1. **Calibration** – The system learns YOUR normal eye openness *and* head pitch angle.
    2. **Detection** – Monitors deviations from YOUR baseline:
       - 👁️ Eye-closure drowsiness (Eye Aspect Ratio drops)
       - 😴 Head droop (chin falling toward chest — pitch drops below your baseline)
       - ↕️ Excessive sideways head tilt / yaw
    3. **Alerts** – Each alert type is shown separately so you know exactly what triggered.
    """)
    
    if 'detector' not in st.session_state:
        st.session_state.detector = PersonalizedDrowsinessDetector()
    if 'calibrated' not in st.session_state:
        st.session_state.calibrated = False
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        
        if st.session_state.calibrated:
            st.success("✅ System Calibrated")
            ub = st.session_state.detector.user_baseline
            if ub:
                st.metric("EAR Threshold",        f"{ub['ear_threshold']:.3f}")
                st.metric("Baseline EAR",          f"{ub['mean_ear']:.3f}")
                st.metric("Baseline Pitch",        f"{ub['baseline_pitch']:.1f}°")
                st.metric("Droop Trigger Below",   f"{ub['pitch_droop_threshold']:.1f}°")
                st.metric("Baseline Yaw",          f"{ub['baseline_yaw']:.1f}°")
                st.metric("Left Tilt Trigger",     f"{ub['yaw_left_threshold']:.1f}°")
                st.metric("Right Tilt Trigger",    f"{ub['yaw_right_threshold']:.1f}°")
        else:
            st.warning("⚠️ Not Calibrated")
        
        st.markdown("---")
        st.markdown("**EAR** – Eye Aspect Ratio: lower = more closed.")
        st.markdown("**Pitch** – Head up/down angle. A significant drop = head droop.")
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Start Calibration (10s)", disabled=st.session_state.calibrated):
            success = st.session_state.detector.calibrate(duration=10)
            if success:
                st.session_state.calibrated = True
                st.rerun()
    
    with col2:
        if st.button("🔄 Recalibrate"):
            st.session_state.detector = PersonalizedDrowsinessDetector()
            st.session_state.calibrated = False
            st.rerun()
    
    if st.session_state.calibrated:
        st.markdown("---")
        if st.button("▶️ Start Drowsiness Detection", type="primary"):
            st.session_state.detector.detect_drowsiness()
    else:
        st.info("👆 Please complete calibration before starting detection.")
    
    with st.expander("ℹ️ How to use this system"):
        st.markdown("""
        **Step 1: Calibration**
        - Click "Start Calibration"
        - Look at the camera naturally for 10 seconds
        - Keep your eyes open normally (don't force them wide)
        - Maintain a comfortable, **upright** posture — this sets your head-pitch baseline

        **Step 2: Detection**
        - Click "Start Drowsiness Detection"
        - The system monitors three signals in real time:
          - 👁️ **Eye closure** – EAR drops below your personal threshold for > 0.67 s
          - 😴 **Head droop** – pitch falls ≥ 8° (or 2.5 std) below your calibrated baseline for > 0.5 s
          - ↕️ **Side tilt** – yaw exceeds ±20°
        - Press "Stop Detection" when done

        **Why personalization matters:**
        - Different people have different natural head angles and eye shapes
        - Calibrating to YOU avoids false alarms and missed detections
        """)


if __name__ == "__main__":
    main()
