import gradio as gr
import speech_recognition as sr
from gtts import gTTS
import os
import cv2
import mediapipe as mp
import io
from PIL import Image

# Mediapipe setup for hand gesture detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Mapping of detected gestures to their meanings
gesture_mapping = {
    "Thumbs Up": "Good job",
    "Open Palm": "Hello",
    "Peace Sign": "Victory",
    "one": "success"
}

# Speech-to-Text Function
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("Listening...")
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
        #os.system("start response.mp3")  # Use 'open' for Mac, 'xdg-open' for Linux
        os.system("open response.mp3")

        return "Audio generated and played successfully."
    except Exception as e:
        return f"Error: {str(e)}"

# Object Detection Function
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
    # Get the landmark positions
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Thumbs Up condition
    if thumb_tip.y < thumb_ip.y and index_tip.y > thumb_tip.y:
        return "Thumbs Up"
    
    # Open Palm condition (all fingers spread out)
    if (index_tip.y < thumb_tip.y and
        middle_tip.y < thumb_tip.y and
        ring_tip.y < thumb_tip.y and
        pinky_tip.y < thumb_tip.y):
        return "Open Palm"
    
    # Peace Sign condition (index and middle fingers raised)
    if index_tip.y < thumb_tip.y and middle_tip.y < index_tip.y:
        return "Peace Sign"

    return "Unknown Gesture"

# Gesture detection and frame streaming
def video_generator():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and process with Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        detection_results = "No gesture detected"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = interpret_gesture(hand_landmarks)
                if gesture in gesture_mapping:
                    detection_results = f"Gesture: {gesture} - Meaning: {gesture_mapping[gesture]}"
                    break

        # Add detection results as overlay text
        cv2.putText(frame, detection_results, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frame to PIL Image
        _, buffer = cv2.imencode('.jpg', frame)
        pil_image = Image.open(io.BytesIO(buffer))

        # Yield both the PIL image and the detection results (gesture text)
        yield pil_image, detection_results

    cap.release()
    cv2.destroyAllWindows()

# Wrapper for Gradio to process video
def process_video():
    for frame, gesture_text in video_generator():
        return frame, gesture_text

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Assistive Tool for Individuals with Disabilities")

    # Speech-to-Text
    with gr.Row():
        gr.Markdown("### Speech-to-Text")
        stt_button = gr.Button("Start Listening")
        stt_output = gr.Textbox(label="Recognized Speech")
        stt_button.click(speech_to_text, inputs=None, outputs=stt_output)

    # Text-to-Speech
    with gr.Row():
        gr.Markdown("### Text-to-Speech")
        tts_input = gr.Textbox(label="Enter Text for Audio")
        tts_button = gr.Button("Generate Audio")
        tts_output = gr.Textbox(label="Status")
        tts_button.click(text_to_speech, inputs=tts_input, outputs=tts_output)

    # Object Detection
    with gr.Row():
        gr.Markdown("### Object Detection")
        detect_button = gr.Button("Detect Objects")
        detect_output = gr.Textbox(label="Status")
        detect_button.click(detect_objects, inputs=None, outputs=detect_output)

    # Sign Language Detection
    with gr.Row():
        gr.Markdown("### Sign Language Detection")
        start_button = gr.Button("Start Sign Language Input")
        video_feed = gr.Image(label="Camera Feed", type="pil", streaming=True)
        sign_output = gr.Textbox(label="Detected Gesture and Meaning")

        # Start button will trigger video feed and gesture detection
        start_button.click(process_video, inputs=None, outputs=[video_feed, sign_output])

# Launch Gradio
demo.launch(share=True, inline=True)