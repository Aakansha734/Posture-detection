import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
from fpdf import FPDF
import os
import json

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Path to the JSON file storing user data
USER_DATA_FILE = "users.json"

# Load user data from JSON file
def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    return {}

# Save user data to JSON file
def save_users(users):
    with open(USER_DATA_FILE, "w") as file:
        json.dump(users, file)

# Dummy user database (loaded from JSON)
users = load_users()

def login_page():
    st.title("AgeWell Motion – Stay active, stay strong!")
    st.subheader("Login or Signup")
    
    choice = st.radio("Select an option", ("Login", "Signup"))
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if choice == "Signup":
        if st.button("Signup"):
            if username in users:
                st.error("Username already exists!")
            else:
                users[username] = {"password": password}
                save_users(users)
                st.success("Signup successful! Please login.")
    else:
        if st.button("Login"):
            if username in users and users[username]["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid credentials!")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

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

    # Posture feedback and YouTube links for correction
  
    if spine_angle > 160 and neck_angle > 150 and leg_angle > 170:
        feedback = ("✅ **Good posture!**\n\n"
        "🔹 **Mindful Awareness Tip:**\n"
        "- Be present: Regularly check your posture throughout the day.\n"
        "- Focus especially when sitting/standing for long periods."
    )
        video_links = [
            "https://www.youtube.com/watch?v=ShnXIqAa3BU",
            "https://www.youtube.com/watch?v=OedABQhP36g",
            "https://www.youtube.com/watch?v=ml9ik3htY_w"

        ]
    elif 140 <= spine_angle <= 160 or 130 <= neck_angle <= 150:
        feedback = (
        "⚠️ **Slightly slouched posture detected!**\n\n"
        "🔹 **Quick fixes:**\n"
        "- Straighten your back and align your ears with your shoulders.\n"
        "- Take a 30-second stretch break every 30 minutes.\n"
        "- Adjust your chair/desk height if seated."
    )
        video_links = [
            "https://www.youtube.com/watch?v=GbGSvAEkE68",
            "https://www.youtube.com/watch?v=JGLR46VGJWQ&t=65s&pp=2AFBkAIB",
            "https://www.youtube.com/watch?v=OMoS7RHua9I&pp=0gcJCfcAhR29_xXO",
            "https://www.youtube.com/watch?v=98injdf2Qso"
            

        ]
    else:
        feedback = (
        "❌ **Bad posture alert!**\n\n"
        "🔹 **Urgent actions needed:**\n"
        "- Stop and reset: Stand up, roll your shoulders back.\n"
        "- Try these corrective exercises (see videos below).\n"
        "- Consider ergonomic adjustments (e.g., lumbar support)."
    )
        video_links = [
            "https://www.youtube.com/watch?v=YAUGMT0_PiE",
            "https://www.youtube.com/watch?v=F_JxvkeFQ78",
            "https://www.youtube.com/watch?v=XxSgdX7lX6E",
            "https://www.youtube.com/watch?v=in9ubCilsT8",
            "https://www.youtube.com/watch?v=g1exhYAKyuE"

        ]
    return feedback, video_links

def generate_report(user_name, age, weight, height, history, posture_feedback, video_links):
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
    pdf.ln(10)
    
    pdf.cell(200, 10, "Recommended YouTube Videos for Posture Correction:", ln=True)
    for link in video_links:
        pdf.cell(200, 10, f"- {link}", ln=True)

    file_path = f"reports.pdf"
    pdf.output(file_path)
    return file_path

def process_video(video_source, duration=10):
    cap = cv2.VideoCapture(video_source)
    start_time = time.time()
    posture_feedback = ""
    video_links = []
    
    st_frame = st.empty()  # Placeholder for displaying the video feed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (time.time() - start_time > duration):
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        result = pose.process(frame_rgb)  # Process frame with MediaPipe
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            posture_feedback, video_links = evaluate_posture(landmarks)
            mp_drawing.draw_landmarks(frame_rgb, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        st_frame.image(frame_rgb, channels="RGB")  # Display the frame in Streamlit

    cap.release()
    return posture_feedback, video_links


def results_page():
    st.title("AgeWell Motion – Stay active, stay strong!")
    st.header("Analysis Results")
    
    if "report_content" in st.session_state and "report_path" in st.session_state:
        st.subheader("Here are your analysis results based on the provided data.")
        
        # Add user details dynamically
        age = st.session_state["user_details"]["age"]
        weight = st.session_state["user_details"]["weight"]
        height = st.session_state["user_details"]["height"]
        medical_condition = st.session_state["user_details"]["medical_condition"]
        
        # Display user details
        st.write(f"**Age:** {age} years")
        st.write(f"**Weight:** {weight} kg")
        st.write(f"**Height:** {height} cm")
        st.write(f"**Medical Condition:** {medical_condition}")
        
        # Display findings based on attributes
        st.subheader("Summary of Findings:")
        st.write(f"**Findings based on the provided data**:")
        st.write(f"- **Height:** {height} cm")
        st.write(f"- **Weight:** {weight} kg")
        st.write(f"- **Age:** {age} years")
        
        # Include the posture feedback (already stored in session)
        st.write(f"**Posture Feedback:** {st.session_state['report_content']}")
        
        # Display YouTube video links
        st.subheader("Recommended YouTube Videos for Posture Correction:")
        for link in st.session_state["video_links"]:
            st.write(f"- [Watch Video]({link})")
        
        # Provide the option to download the report
        with open(st.session_state["report_path"], "rb") as f:
            st.download_button("Download Report", f, file_name=st.session_state["report_path"])
    else:
        st.warning("No report available. Please perform an analysis first.")


def main_app():
    st.title("AgeWell Motion – Stay active, stay strong!")
    st.header("Capture Video for Posture Analysis")

    # Add an option to choose between video upload and camera capture
    option = st.radio("Choose your video input method", ("Upload Video", "Capture from Camera"))

    # User Details Section (Always shown before uploading video or capturing)
    st.subheader("Please enter your details:")
    
    # Inputs for the patient's details
    age = st.number_input("Age", min_value=0, max_value=120, value=65)
    weight = st.number_input("Weight (kg)", min_value=0, value=70)
    height = st.number_input("Height (cm)", min_value=0, value=170)
    medical_condition = st.text_input("Medical Condition (if any)", value="None")

    # Store the user details in the session state for later use
    st.session_state["user_details"] = {
        "age": age,
        "weight": weight,
        "height": height,
        "medical_condition": medical_condition
    }

    if option == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_video_path = temp_file.name

            st.video(temp_video_path)

            st.write("Processing video...")
            posture_feedback, video_links = process_video(temp_video_path)

            st.session_state["report_content"] = posture_feedback
            st.session_state["video_links"] = video_links
            st.session_state["report_path"] = generate_report(
                st.session_state["username"], age, weight, height, medical_condition, posture_feedback, video_links
            )

            st.success("Analysis completed! Redirecting to results page...")
            st.rerun()

    elif option == "Capture from Camera":
        st.write("Click the button to start recording for 10 seconds.")

        # Start recording button
        if st.button("Start Recording"):
            cap = cv2.VideoCapture(0)  # Open webcam (index 0 for default webcam)
            frame_count = 0
            start_time = time.time()
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret or (time.time() - start_time) > 10:  # Capture for 10 seconds
                    break

                frames.append(frame)
                frame_count += 1

            cap.release()

            # Convert frames to a temporary video file for further processing
            temp_video_path = tempfile.mktemp(suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (640, 480))

            for frame in frames:
                out.write(frame)
            out.release()

            st.video(temp_video_path)

            # Process the captured video
            posture_feedback, video_links = process_video(temp_video_path)

            st.session_state["report_content"] = posture_feedback
            st.session_state["video_links"] = video_links
            st.session_state["report_path"] = generate_report(
                st.session_state["username"], age, weight, height, medical_condition, posture_feedback, video_links
            )

            st.success("Analysis completed! Redirecting to results page...")
            st.rerun()


# Ensure correct page navigation
if "logged_in" not in st.session_state:
    login_page()
elif "report_content" in st.session_state:
    results_page()
else:
    main_app()