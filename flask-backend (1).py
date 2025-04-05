from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import threading
import time
import os
import json
from datetime import datetime
from ultralytics import YOLO
import base64

app = Flask(__name__, static_folder='static', template_folder='templates')

# Global variables
camera = None
detection_thread = None
is_detection_active = False
frame_count = 0
detected_violations = []
model = None
detection_settings = {
    'detection_type': 'all',
    'sensitivity': 7,
    'threshold': 0.85
}

# Load YOLOv8 model
def load_model():
    global model
    model = YOLO('yolov8n.pt')  # Replace with your trained model
    print("Model loaded successfully")

# Initialize license plate cascade
try:
    license_plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
    if license_plate_cascade.empty():
        print("Warning: Could not load license plate cascade classifier")
except Exception as e:
    print(f"Error loading cascade classifier: {e}")
    license_plate_cascade = None

# Initialize camera
def initialize_camera(source=0):
    global camera
    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return False
    return True

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def get_current_frame():
    global camera
    if camera is not None:
        ret, frame = camera.read()
        if ret:
            return frame
    return None

# Function to detect traffic violations
def detect_violations(frame):
    global model, detection_settings, license_plate_cascade
    
    if frame is None or model is None:
        return frame, []
    
    # Convert threshold from percentage to decimal
    conf_threshold = detection_settings['threshold'] / 100
    
    # Process frame with YOLOv8
    results = model(frame, conf=conf_threshold)
    
    # Get detections
    detections = []
    annotated_frame = results[0].plot()
    
    # Process based on detection type
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Get class name
            class_name = result.names[cls]
            
            # Check if this is the type of violation we're looking for
            if (detection_settings['detection_type'] == 'all' or 
                (detection_settings['detection_type'] == 'speed' and class_name == 'car') or
                (detection_settings['detection_type'] == 'redlight' and class_name == 'traffic light') or
                (detection_settings['detection_type'] == 'helmet' and class_name == 'motorcycle')):
                
                # Extract ROI for license plate detection (if this is a vehicle)
                if class_name in ['car', 'truck', 'motorcycle', 'bus']:
                    roi = frame[y1:y2, x1:x2]
                    license_plate = detect_license_plate(roi)
                    
                    # Create a detection object
                    detection = {
                        'type': determine_violation_type(class_name),
                        'confidence': conf,
                        'license_plate': license_plate,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'image': save_violation_image(frame[y1:y2, x1:x2])
                    }
                    
                    detections.append(detection)
    
    return annotated_frame, detections

def determine_violation_type(class_name):
    # In a real implementation, this would use more sophisticated logic
    if class_name == 'car':
        return 'Speeding'
    elif class_name == 'traffic light':
        return 'Red Light'
    elif class_name == 'motorcycle':
        return 'No Helmet'
    else:
        return 'Other'

def detect_license_plate(roi):
    global license_plate_cascade
    if license_plate_cascade is None or roi is None or roi.size == 0:
        return "Unknown"
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Detect license plates
    plates = license_plate_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(plates) > 0:
        # In a real implementation, you would use OCR here
        # For this example, we'll just return a placeholder
        return f"ABC-{np.random.randint(1000, 9999)}"
    
    return "Unknown"

def save_violation_image(roi):
    if roi is None or roi.size == 0:
        return ""
    
    # Create directory if not exists
    if not os.path.exists("static/violations"):
        os.makedirs("static/violations")
    
    # Generate a filename
    filename = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000)}.jpg"
    filepath = f"static/violations/{filename}"
    
    # Save the image
    cv2.imwrite(filepath, roi)
    
    return filepath

def detection_loop():
    global is_detection_active, camera, detected_violations, frame_count
    
    while is_detection_active:
        frame = get_current_frame()
        if frame is not None:
            processed_frame, new_detections = detect_violations(frame)
            
            # Add new detections to the list
            for detection in new_detections:
                detected_violations.append(detection)
                
                # Keep only the last 100 detections
                if len(detected_violations) > 100:
                    detected_violations.pop(0)
            
            # Store processed frame for streaming
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_count += 1

        time.sleep(0.03)  # 30 FPS target

def generate_frames():
    global frame_count
    last_frame_count = 0
    
    while True:
        # Wait until we have a new frame
        while frame_count == last_frame_count:
            time.sleep(0.01)
            if not is_detection_active:
                break
                
        if not is_detection_active:
            break
            
        last_frame_count = frame_count
        frame = get_current_frame()
        
        if frame is not None:
            _, processed_frame = detect_violations(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        
        time.sleep(0.03)

# API Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start-detection', methods=['POST'])
def start_detection():
    global is_detection_active, detection_thread
    
    if is_detection_active:
        return jsonify({'status': 'already_running'})
    
    # Initialize camera and model if not already done
    if camera is None:
        if not initialize_camera():
            return jsonify({'status': 'error', 'message': 'Failed to initialize camera'})
    
    if model is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Failed to load model: {str(e)}'})
    
    # Start detection thread
    is_detection_active = True
    detection_thread = threading.Thread(target=detection_loop)
    detection_thread.daemon = True
    detection_thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/stop-detection', methods=['POST'])
def stop_detection():
    global is_detection_active, detection_thread
    
    is_detection_active = False
    
    if detection_thread is not None:
        detection_thread.join(timeout=1.0)
        detection_thread = None
    
    # Don't release the camera here to allow for quick restart
    
    return jsonify({'status': 'stopped'})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    global detection_settings
    
    data = request.json
    if data:
        for key, value in data.items():
            if key in detection_settings:
                detection_settings[key] = value
    
    return jsonify({'status': 'updated', 'settings': detection_settings})

@app.route('/api/violations', methods=['GET'])
def get_violations():
    global detected_violations
    return jsonify(detected_violations)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    global detected_violations
    
    # Calculate stats
    today_violations = sum(1 for v in detected_violations if 
                          v['timestamp'].startswith(datetime.now().strftime("%Y-%m-%d")))
    
    stats = {
        'today_violations': today_violations,
        'total_detected': len(detected_violations)
    }
    
    return jsonify(stats)

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    # Save the uploaded file
    upload_path = 'uploads/'
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
        
    filepath = os.path.join(upload_path, file.filename)
    file.save(filepath)
    
    # Process the video (this would typically be done in a separate thread)
    # For this example, we'll just return success
    return jsonify({'status': 'success', 'message': 'Video uploaded successfully'})

if __name__ == '__main__':
    # Load model on startup
    load_model()
    app.run(debug=True)
