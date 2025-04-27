'''import cv2
import dlib
from scipy.spatial import distance as dist
import time
import pyttsx3  # Text-to-speech library
import bluetooth  # For Bluetooth communication

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indices for eye landmarks
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to speak the action
def speak_action(action):
    engine.say(action)
    engine.runAndWait()

# Function to connect to Bluetooth module
def connect_bluetooth():
    try:
        bd_addr = "98:D3:31:FC:32:B3"  # Replace with your Bluetooth module's MAC address
        port = 1  # RFCOMM port
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((bd_addr, port))
        print("Bluetooth connected!")
        return sock
    except Exception as e:
        print(f"Bluetooth connection failed: {e}")
        return None

# Constants
EAR_THRESHOLD = 0.25  # Threshold for eye closure
BLINK_THRESHOLD = 0.2  # Threshold to detect a blink
BLINK_COUNT_TIME = 1.5  # Time window to count blinks (in seconds)
LONG_CLOSE_TIME = 2.0  # Time to consider long eye closure (in seconds)

# State variables
blink_count = 0
last_blink_time = time.time()
eyes_closed_time = None

# Connect to Bluetooth
sock = connect_bluetooth()

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE_POINTS]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE_POINTS]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0  # Average EAR for both eyes

        # Check if eyes are closed
        if ear < EAR_THRESHOLD:
            if eyes_closed_time is None:
                eyes_closed_time = time.time()
            else:
                if time.time() - eyes_closed_time > LONG_CLOSE_TIME:
                    print("Stop")
                    cv2.putText(frame, "Command: Stop", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    speak_action("Stop, stop")
                    if sock:
                        sock.send("S")  # Send 'S' for stop
                    eyes_closed_time = None
        else:
            eyes_closed_time = None

        # Detect blinks
        if ear < BLINK_THRESHOLD:
            if time.time() - last_blink_time > 0.3:  # Debounce multiple detections
                blink_count += 1
                last_blink_time = time.time()

        # Check blink count within the time window
        if time.time() - last_blink_time > BLINK_COUNT_TIME:
            if blink_count == 1:
                print("Move Forward")
                cv2.putText(frame, "Command: Move Forward", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                speak_action("Moving Forward")
                if sock:
                    sock.send("F")  # Send 'F' for move forward
            elif blink_count == 2:
                print("Turn Left")
                cv2.putText(frame, "Command: Turn Left", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                speak_action(" Moving Left,left")
                if sock:
                    sock.send("L")  # Send 'L' for turn left
            elif blink_count == 3:
                print("Turn Right")
                cv2.putText(frame, "Command: Turn Right", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                speak_action(" Moving Right, right")
                if sock:
                    sock.send("R")  # Send 'R' for turn right
            blink_count = 0  # Reset blink count

        # Display EAR and blink count on the frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Eye Controlled Wheelchair', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if sock:
    sock.close()'''
    
import cv2
import dlib
from scipy.spatial import distance as dist
import time
import pyttsx3  # Text-to-speech library
import bluetooth  # For Bluetooth communication

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load face detector and landmark predictor
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except Exception as e:
    print("Error loading dlib model:", e)
    exit()

# Indices for eye landmarks
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to speak the action
def speak_action(action):
    engine.say(action)
    engine.runAndWait()

# Function to connect to Bluetooth module
def connect_bluetooth():
    try:
        bd_addr = "98:D3:31:FC:32:B3"  # Replace with your Bluetooth module's MAC address
        port = 1  # RFCOMM port
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((bd_addr, port))
        print("Bluetooth connected!")
        return sock
    except Exception as e:
        print(f"Bluetooth connection failed: {e}")
        return None

# Constants
EAR_THRESHOLD = 0.25  # Threshold for eye closure
BLINK_THRESHOLD = 0.2  # Threshold to detect a blink
BLINK_COUNT_TIME = 1.5  # Time window to count blinks (in seconds)
LONG_CLOSE_TIME = 2.0  # Time to consider long eye closure (in seconds)

# State variables
blink_count = 0
last_blink_time = time.time()
eyes_closed_time = None

# Connect to Bluetooth
sock = connect_bluetooth()

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray is None or gray.dtype != 'uint8':
        print("Invalid grayscale image")
        continue

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE_POINTS]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE_POINTS]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0  # Average EAR for both eyes

        # Check if eyes are closed
        if ear < EAR_THRESHOLD:
            if eyes_closed_time is None:
                eyes_closed_time = time.time()
            else:
                if time.time() - eyes_closed_time > LONG_CLOSE_TIME:
                    print("Stop")
                    cv2.putText(frame, "Command: Stop", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    speak_action("Stop, stop")
                    if sock:
                        sock.send("S")  # Send 'S' for stop
                    eyes_closed_time = None
        else:
            eyes_closed_time = None

        # Detect blinks
        if ear < BLINK_THRESHOLD:
            if time.time() - last_blink_time > 0.3:  # Debounce multiple detections
                blink_count += 1
                last_blink_time = time.time()

        # Check blink count within the time window
        if time.time() - last_blink_time > BLINK_COUNT_TIME:
            if blink_count == 1:
                print("Move Forward")
                cv2.putText(frame, "Command: Move Forward", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                speak_action("Moving Forward")
                if sock:
                    sock.send("F")  # Send 'F' for move forward
            elif blink_count == 2:
                print("Turn Left")
                cv2.putText(frame, "Command: Turn Left", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                speak_action(" Moving Left, left")
                if sock:
                    sock.send("L")  # Send 'L' for turn left
            elif blink_count == 3:
                print("Turn Right")
                cv2.putText(frame, "Command: Turn Right", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                speak_action(" Moving Right, right")
                if sock:
                    sock.send("R")  # Send 'R' for turn right
            blink_count = 0  # Reset blink count

        # Display EAR and blink count on the frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Eye Controlled Wheelchair', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if sock:
    sock.close()

