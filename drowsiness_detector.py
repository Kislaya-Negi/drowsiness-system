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
        
        # Head pose landmarks (nose tip, chin, left eye, right eye, left mouth, right mouth)
        self.HEAD_POSE_LANDMARKS = [1, 152, 263, 33, 61, 291]
        
        # User baseline data
        self.user_baseline = None
        self.calibration_data = {
            'ear_values': [],
            'blink_durations': [],
            'head_poses': []
        }
        
        # Real-time tracking
        self.ear_history = deque(maxlen=30)  # Last 1 second at 30fps
        self.blink_counter = 0
        self.blink_start_time = None
        self.total_blinks = 0
        self.drowsiness_score = 0
        
        # Drowsiness thresholds (will be personalized)
        self.ear_threshold = None
        self.blink_duration_threshold = None
        self.consecutive_frames_threshold = 20  # About 0.67 seconds
        self.consecutive_low_ear = 0
        
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for given eye landmarks"""
        # Vertical distances
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        # Horizontal distance
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_head_pose(self, landmarks, image_shape):
        """Calculate head pose angles"""
        h, w = image_shape[:2]
        
        # Get 2D coordinates of key facial landmarks
        face_2d = []
        face_3d = []
        
        for idx in self.HEAD_POSE_LANDMARKS:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
        
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        
        # Camera matrix
        focal_length = w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ])
        
        # Distortion coefficients (assuming no distortion)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, dist_matrix
        )
        
        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)
        
        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        # Get the y rotation degree (head tilt)
        y_rotation = angles[1] * 360
        x_rotation = angles[0] * 360
        
        return x_rotation, y_rotation
    
    def calibrate(self, duration=10):
        """
        Calibrate the system for the current user.
        User should maintain normal, alert driving posture during calibration.
        """
        st.info(f"🎯 Calibration Phase: Please look at the camera naturally for {duration} seconds.")
        st.info("Keep your eyes open normally and maintain a comfortable posture.")
        
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
                
                # Extract eye landmarks
                left_eye = [(landmarks[i].x, landmarks[i].y) for i in self.LEFT_EYE]
                right_eye = [(landmarks[i].x, landmarks[i].y) for i in self.RIGHT_EYE]
                
                # Calculate EAR
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                self.calibration_data['ear_values'].append(avg_ear)
                
                # Get head pose
                x_rot, y_rot = self.get_head_pose(landmarks, frame.shape)
                self.calibration_data['head_poses'].append((x_rot, y_rot))
                
                # Display frame
                cv2.putText(frame, f"Calibrating... {int(duration - (time.time() - start_time))}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                frame_count += 1
            
            # Update progress
            progress = (time.time() - start_time) / duration
            progress_bar.progress(min(progress, 1.0))
            
            # Display frame in Streamlit
            calibration_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        cap.release()
        progress_bar.empty()
        calibration_placeholder.empty()
        
        # Calculate personalized thresholds
        if len(self.calibration_data['ear_values']) > 0:
            ear_values = np.array(self.calibration_data['ear_values'])
            
            # Set threshold as mean - 1.5 * std (accounts for natural variation)
            self.ear_threshold = np.mean(ear_values) - 1.5 * np.std(ear_values)
            
            # Ensure threshold is reasonable
            self.ear_threshold = max(0.15, min(0.25, self.ear_threshold))
            
            self.user_baseline = {
                'mean_ear': np.mean(ear_values),
                'std_ear': np.std(ear_values),
                'ear_threshold': self.ear_threshold,
                'calibration_frames': frame_count
            }
            
            st.success(f"✅ Calibration Complete! Your personalized EAR threshold: {self.ear_threshold:.3f}")
            st.info(f"Baseline EAR: {self.user_baseline['mean_ear']:.3f} ± {self.user_baseline['std_ear']:.3f}")
            return True
        else:
            st.error("❌ Calibration failed. Please ensure your face is visible to the camera.")
            return False
    
    def detect_drowsiness(self):
        """
        Real-time drowsiness detection using personalized thresholds
        """
        if self.user_baseline is None:
            st.error("Please complete calibration first!")
            return
        
        st.info("🚗 Drowsiness Detection Active. Press 'Stop' button to end.")
        
        cap = cv2.VideoCapture(0)
        
        # Streamlit placeholders
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        stop_button = st.button("Stop Detection")
        
        alert_active = False
        alert_start_time = None
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            drowsy = False
            status_text = "ALERT ✓"
            status_color = (0, 255, 0)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Extract eye landmarks
                left_eye = [(landmarks[i].x, landmarks[i].y) for i in self.LEFT_EYE]
                right_eye = [(landmarks[i].x, landmarks[i].y) for i in self.RIGHT_EYE]
                
                # Calculate EAR
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                self.ear_history.append(avg_ear)
                
                # Check for drowsiness
                if avg_ear < self.ear_threshold:
                    self.consecutive_low_ear += 1
                    
                    if self.consecutive_low_ear >= self.consecutive_frames_threshold:
                        drowsy = True
                        status_text = "⚠️ DROWSINESS DETECTED ⚠️"
                        status_color = (0, 0, 255)
                        
                        if not alert_active:
                            alert_active = True
                            alert_start_time = time.time()
                else:
                    self.consecutive_low_ear = 0
                    alert_active = False
                
                # Get head pose
                x_rot, y_rot = self.get_head_pose(landmarks, frame.shape)
                
                # Check for excessive head tilting (possible nodding off)
                if abs(x_rot) > 20 or abs(y_rot) > 20:
                    status_text = "⚠️ HEAD TILTING DETECTED ⚠️"
                    status_color = (0, 165, 255)
                    drowsy = True
                
                # Draw eye landmarks
                h, w = frame.shape[:2]
                for eye in [left_eye, right_eye]:
                    eye_points = np.array([(int(x * w), int(y * h)) for x, y in eye], np.int32)
                    cv2.polylines(frame, [eye_points], True, (0, 255, 0), 1)
                
                # Display metrics on frame
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Threshold: {self.ear_threshold:.2f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, status_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Visual alert
                if drowsy:
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            
            # Display frame
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Display status
            if drowsy:
                status_placeholder.error(f"🚨 {status_text}")
            else:
                status_placeholder.success(f"✅ {status_text}")
            
            # Display metrics
            if len(self.ear_history) > 0:
                metrics_placeholder.metric(
                    "Current EAR", 
                    f"{np.mean(list(self.ear_history)):.3f}",
                    delta=f"{np.mean(list(self.ear_history)) - self.user_baseline['mean_ear']:.3f}"
                )
        
        cap.release()
        video_placeholder.empty()
        status_placeholder.empty()
        metrics_placeholder.empty()
        st.success("Detection stopped.")


def main():
    st.set_page_config(page_title="Personalized Drowsiness Detector", layout="wide")
    
    st.title("🚗 Personalized Drowsiness Detection System")
    st.markdown("""
    ### An Unbiased, Adaptive Drowsiness Detection System
    
    This system eliminates bias by **personalizing to your unique facial features** rather than 
    using absolute thresholds that may not work for everyone.
    
    **How it works:**
    1. **Calibration**: The system learns YOUR normal eye appearance and behavior
    2. **Detection**: Monitors deviations from YOUR baseline to detect drowsiness
    3. **Alerts**: Warns you when drowsiness patterns are detected
    """)
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = PersonalizedDrowsinessDetector()
    
    if 'calibrated' not in st.session_state:
        st.session_state.calibrated = False
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        
        if st.session_state.calibrated:
            st.success("✅ System Calibrated")
            if st.session_state.detector.user_baseline:
                st.metric("Your EAR Threshold", 
                         f"{st.session_state.detector.user_baseline['ear_threshold']:.3f}")
                st.metric("Baseline EAR", 
                         f"{st.session_state.detector.user_baseline['mean_ear']:.3f}")
        else:
            st.warning("⚠️ Not Calibrated")
        
        st.markdown("---")
        st.markdown("**About EAR (Eye Aspect Ratio)**")
        st.markdown("A metric that measures eye openness. Lower values indicate more closed eyes.")
    
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
    
    # Information footer
    with st.expander("ℹ️ How to use this system"):
        st.markdown("""
        **Step 1: Calibration**
        - Click "Start Calibration"
        - Look at the camera naturally for 10 seconds
        - Keep your eyes open normally (don't force them wide)
        - Maintain a comfortable, alert posture
        
        **Step 2: Detection**
        - Click "Start Drowsiness Detection"
        - The system will monitor you in real-time
        - You'll get alerts if drowsiness is detected
        - Press "Stop Detection" when done
        
        **Why personalization matters:**
        - Different people have different eye shapes and sizes
        - What looks "drowsy" varies between individuals
        - This system adapts to YOUR unique features
        - Eliminates bias that affects traditional systems
        """)


if __name__ == "__main__":
    main()