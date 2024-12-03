import gradio as gr
import speech_recognition as sr
from gtts import gTTS
import os
import cv2
import mediapipe as mp
import io
from PIL import Image
import tkinter as tk

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

# Braille dictionary: maps Braille pattern (6-bit) to corresponding letter
braille_dict = {
    '100000': 'a', '101000': 'b', '110000': 'c', '110100': 'd', '100100': 'e',
    '111000': 'f', '111100': 'g', '101100': 'h', '011000': 'i', '011100': 'j',
    '100010': 'k', '101010': 'l', '110010': 'm', '110110': 'n', '100110': 'o',
    '111010': 'p', '111110': 'q', '101110': 'r', '011010': 's', '011110': 't',
    '100011': 'u', '101011': 'v', '011101': 'w', '110011': 'x', '110111': 'y',
    '100111': 'z'
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
        os.system("start response.mp3")  # Use 'open' for Mac, 'xdg-open' for Linux
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

# Braille Input System (Tkinter integration)
class BrailleInputApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Braille Input via Trackpad")
        
        # Store the Braille pattern
        self.braille_input = ""
        
        # Create Braille cell buttons (2x3 grid)
        self.buttons = []
        for i in range(2):
            row_buttons = []
            for j in range(3):
                button = tk.Button(root, text=".", width=10, height=5,
                                   bg="lightgray", command=lambda r=i, c=j: self.toggle_dot(r, c))
                button.grid(row=i, column=j)
                row_buttons.append(button)
            self.buttons.append(row_buttons)
        
        # Add a Convert button
        self.convert_button = tk.Button(root, text="Convert to Text", width=20, height=2, command=self.convert_braille_to_text)
        self.convert_button.grid(row=2, columnspan=3)
        
        # Add a Label to show the converted text
        self.output_label = tk.Label(root, text="Converted Text: ", font=("Arial", 14))
        self.output_label.grid(row=3, columnspan=3)
        
    def toggle_dot(self, row, col):
        """Toggle the state of the button (dot on/off)"""
        button = self.buttons[row][col]
        if button['bg'] == 'lightgray':
            button['bg'] = 'blue'  # Activated dot
        else:
            button['bg'] = 'lightgray'  # Deactivated dot
        self.update_braille_input()
    
    def update_braille_input(self):
        """Update the Braille input string based on the active dots"""
        self.braille_input = ""
        for i in range(2):
            for j in range(3):
                button = self.buttons[i][j]
                self.braille_input += '1' if button['bg'] == 'blue' else '0'
    
    def convert_braille_to_text(self):
        """Convert the Braille pattern to text and display it"""
        if self.braille_input in braille_dict:
            text = braille_dict[self.braille_input]
            self.output_label.config(text=f"Converted Text: {text}")
            self.speak_text(text)
        else:
            self.output_label.config(text="Converted Text: Invalid Braille input")
    
    def speak_text(self, text):
        """Convert text to speech and play it"""
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")
        os.system("start output.mp3")  # For Windows, use 'start'. On Mac, use 'open' and on Linux 'xdg-open'

# Set up Gradio interface with speech-to-text, video stream, and Braille input
def create_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Button("Start Speech-to-Text", variant="primary").click(speech_to_text, outputs="text")
            gr.Button("Start Object Detection").click(detect_objects, outputs="text")
            gr.Button("Start Gesture Detection").click(process_video, outputs=[gr.Image(), gr.Text()])
            
        demo.launch()

if __name__ == "__main__":
    # Create the Tkinter window for Braille Input
    root = tk.Tk()
    app = BrailleInputApp(root)
    root.mainloop()