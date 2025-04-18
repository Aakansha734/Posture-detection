import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
from fpdf import FPDF
import os

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

    spine_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    neck_angle = calculate_angle(left_ear, left_shoulder, left_hip)
    leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

    # Posture Analysis
    if spine_angle > 160 and neck_angle > 150 and leg_angle > 170:
        return "Good posture! ✅", "https://www.youtube.com/watch?v=ml9ik3htY_w"
    elif 140 <= spine_angle <= 160 or 130 <= neck_angle <= 150:
        return "Slightly slouched. ⚠️ Try to straighten your back.", "https://www.youtube.com/watch?v=GbGSvAEkE68"
    else:
        return "Bad posture detected! ❌ Consider corrective exercises.", "https://www.youtube.com/watch?v=F_JxvkeFQ78"

# Generate PDF Report
def generate_report(user_name, age, weight, height, history, posture_feedback):
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
    
    pdf.cell(200, 10, f"Posture Feedback: {posture_feedback.encode('latin-1', 'ignore').decode('latin-1')}", ln=True)

    
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

def process_video(video_source, duration=30):
    cap = cv2.VideoCapture(video_source)
    start_time = time.time()
    frame_placeholder = st.empty()
    feedback_placeholder = st.empty()
    posture_feedback = ""
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (time.time() - start_time > duration):
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = result.pose_landmarks.landmark
            posture_feedback, youtube_link = evaluate_posture(landmarks)
            feedback_placeholder.subheader(posture_feedback)
            st.video(youtube_link)
        
        frame_placeholder.image(frame, channels="RGB")
    
    cap.release()
    return posture_feedback

if input_method == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        posture_feedback = process_video(temp_file.name)
        
        report_path = generate_report(user_name, age, weight, height, history, posture_feedback)
        st.success("Posture report generated!")
        with open(report_path, "rb") as f:
            st.download_button("Download Report", f, file_name=report_path)

elif input_method == "Use Camera":
    if st.button("Start Analysis for 10 seconds"):
        posture_feedback = process_video(0)
        
        report_path = generate_report(user_name, age, weight, height, history, posture_feedback)
        st.success("Posture report generated!")
        with open(report_path, "rb") as f:
            st.download_button("Download Report", f, file_name=report_path)
