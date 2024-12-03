import gradio as gr
import speech_recognition as sr
from gtts import gTTS
import os
import cv2
import mediapipe as mp
import threading
import time
from PIL import Image
import io

# Mediapipe setup for hand gesture detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Gesture mapping
gesture_mapping = {
    "Thumbs Up": "Good job",
    "Open Palm": "Hello",
    "Peace Sign": "Victory",
    "One": "Success"
}

# Braille dictionary
braille_dict = {
    '100000': 'a', '101000': 'b', '110000': 'c', '110100': 'd', '100100': 'e',
    '111000': 'f', '111100': 'g', '101100': 'h', '011000': 'i', '011100': 'j',
    '100010': 'k', '101010': 'l', '110010': 'm', '110110': 'n', '100110': 'o',
    '111010': 'p', '111110': 'q', '101110': 'r', '011010': 's', '011110': 't',
    '100011': 'u', '101011': 'v', '011101': 'w', '110011': 'x', '110111': 'y',
    '100111': 'z'
}

# Global variables
gesture_detection_running = False
frame_buffer = None
gesture_text = "No gesture detected"

# Speech-to-Text Function
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Error with the recognition service."

# Text-to-Speech Function
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("response.mp3")
        os.system("open response.mp3")  # Use 'open' for Mac, 'xdg-open' for Linux
        return "Audio generated and played successfully."
    except Exception as e:
        return f"Error: {str(e)}"

# Object Detection Function (using YOLO)
def detect_objects():
    try:
        # Load YOLO model
        net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
        classes = []
        with open("yolo/coco.names", "r") as f:
            classes = f.read().strip().split("\n")

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            return "Failed to capture image from camera."

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        detection_results = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]
                if confidence > 0.5:
                    class_name = classes[class_id]
                    detection_results.append(f"Detected: {class_name} with confidence {confidence:.2f}")

        cap.release()
        cv2.destroyAllWindows()

        if detection_results:
            return "\n".join(detection_results)
        else:
            return "No objects detected with sufficient confidence."
    except Exception as e:
        return f"Error: {str(e)}"

# Utility function to interpret hand gestures
def interpret_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    if thumb_tip.y < index_tip.y and middle_tip.y > index_tip.y:
        return "Thumbs Up"
    if index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y:
        return "Peace Sign"
    return "Unknown Gesture"

# Real-Time Gesture Detection
def start_gesture_detection():
    global gesture_detection_running, frame_buffer, gesture_text
    gesture_detection_running = True
    cap = cv2.VideoCapture(0)
    while gesture_detection_running:
        ret, frame = cap.read()
        if not ret:
            gesture_text = "Error accessing camera."
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        gesture_text = "No gesture detected"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture_text = interpret_gesture(hand_landmarks)

        # Save frame to buffer
        frame_buffer = frame

    cap.release()
    cv2.destroyAllWindows()
    return "Gesture detection stopped."

def stop_gesture_detection():
    global gesture_detection_running
    gesture_detection_running = False
    return "Gesture detection stopped."

def video_stream():
    global frame_buffer
    while True:
        if frame_buffer is not None:
            _, buffer = cv2.imencode('.jpg', frame_buffer)
            pil_image = Image.open(io.BytesIO(buffer))
            yield pil_image, gesture_text
        else:
            yield None, "No gesture detected."

# Braille Translation Function
def translate_braille(braille_input):
    if len(braille_input) != 6 or not set(braille_input).issubset({'0', '1'}):
        return "Invalid Braille input. Use a 6-bit binary string (e.g., '101000')."
    return braille_dict.get(braille_input, "No corresponding letter for this pattern.")

# Gradio Interface
def create_interface():
    with gr.Blocks() as demo:
        
       
            
        with gr.Tab("Speech Recognition"):
            speech_to_text_btn = gr.Button("Convert Speech to Text")
            speech_output = gr.Textbox(label="Recognized Speech")
            speech_to_text_btn.click(speech_to_text, outputs=speech_output)

        with gr.Tab("Object Detection"):
            object_detection_btn = gr.Button("Start Object Detection")
            object_detection_output = gr.Textbox(label="Detected Objects")
            object_detection_btn.click(detect_objects, outputs=object_detection_output)

        with gr.Tab("Gesture Recognition"):
            start_gesture_btn = gr.Button("Start Gesture Detection")
            stop_gesture_btn = gr.Button("Stop Gesture Detection")
            video_output = gr.Image(label="Video Stream")
            gesture_output = gr.Textbox(label="Detected Gesture")

            start_gesture_btn.click(start_gesture_detection)
            stop_gesture_btn.click(stop_gesture_detection)
            start_gesture_btn.click(video_stream, outputs=[video_output, gesture_output])

        with gr.Tab("Braille Translation"):
            braille_input = gr.Textbox(label="Enter Braille Code (6-bit binary)", placeholder="e.g., 101000")
            braille_output = gr.Textbox(label="Translated Letter")
            translate_btn = gr.Button("Translate Braille")
            translate_btn.click(translate_braille, inputs=braille_input, outputs=braille_output)

        demo.launch()

if __name__ == "__main__":
    create_interface()