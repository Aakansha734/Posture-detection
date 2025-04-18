import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
from collections import Counter
from fpdf import FPDF

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

# Posture Evaluation Function
def evaluate_posture(landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, 
                landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    # Calculate angles
    spine_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    neck_angle = calculate_angle(left_ear, left_shoulder, left_hip)
    leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

    # Determine posture state
    if spine_angle > 160 and neck_angle > 150 and leg_angle > 170:
        return "Good Posture"
    elif 140 <= spine_angle <= 160 or 130 <= neck_angle <= 150:
        return "Slightly Slouched"
    else:
        return "Bad Posture"

# Generate PDF Report
def generate_report(user_name, age, weight, height, history, most_common_posture):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, f"Posture Analysis Report - {user_name}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.cell(200, 10, f"Age: {age} years", ln=True)
    pdf.cell(200, 10, f"Weight: {weight} kg", ln=True)
    pdf.cell(200, 10, f"Height: {height} cm", ln=True)
    pdf.cell(200, 10, f"Medical History: {history}", ln=True)
    pdf.ln(10)

    # Posture Summary
    pdf.cell(200, 10, f"Most Common Posture: {most_common_posture}", ln=True)

    # Suggested Correction Video Based on Most Frequent Posture
    if most_common_posture == "Good Posture":
        correction_video = "https://www.youtube.com/watch?v=F75pOh1WK3Y"
    elif most_common_posture == "Slightly Slouched":
        correction_video = "https://www.youtube.com/watch?v=2fmZFzC8g6A"
    else:
        correction_video = "https://www.youtube.com/watch?v=4BOTvaRaDjI"  # General posture correction

    pdf.cell(200, 10, f"Recommended Correction: {correction_video}", ln=True)

    file_path = f"{user_name}_posture_report.pdf"
    pdf.output(file_path)
    return file_path

# Streamlit App UI
st.title("Elderly Posture & Fall Detection System for Doctors")
st.sidebar.header("Patient Information")
user_name = st.sidebar.text_input("Name:")
age = st.sidebar.number_input("Age", min_value=40, max_value=100, step=1)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250)
history = st.sidebar.text_area("Medical History (e.g., Arthritis, Back Pain)")

input_method = st.radio("Select Input Method:", ("Upload Video", "Use Camera"))

def process_video(video_source, duration=20):
    cap = cv2.VideoCapture(video_source)
    start_time = time.time()
    frame_placeholder = st.empty()
    
    posture_states = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (time.time() - start_time > duration):
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = result.pose_landmarks.landmark
            posture_state = evaluate_posture(landmarks)
            posture_states.append(posture_state)
        
        frame_placeholder.image(frame, channels="RGB")
    
    cap.release()

    # Find the most common posture state
    if posture_states:
        most_common_posture = Counter(posture_states).most_common(1)[0][0]
    else:
        most_common_posture = "No posture detected"

    return most_common_posture

if input_method == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        most_common_posture = process_video(temp_file.name)
        
        report_path = generate_report(user_name, age, weight, height, history, most_common_posture)
        st.success(f"Posture report generated! Most common posture: {most_common_posture}")
        with open(report_path, "rb") as f:
            st.download_button("Download Report", f, file_name=report_path)

elif input_method == "Use Camera":
    if st.button("Start Analysis for 20 seconds"):
        most_common_posture = process_video(0)
        
        report_path = generate_report(user_name, age, weight, height, history, most_common_posture)
        st.success(f"Posture report generated! Most common posture: {most_common_posture}")
        with open(report_path, "rb") as f:
            st.download_button("Download Report", f, file_name=report_path)
