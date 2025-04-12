import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract  # Replace EasyOCR with Tesseract
from PIL import Image
import os
# os.environ["OPENCV_IO_ENABLE_WINDOWS"] = "1"

# Configure Tesseract path (change this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example

# Load models
model = YOLO('yolov8n.pt')
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Face recognition setup
FACE_DATASET = "faces_dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
FACE_RECOGNITION_THRESHOLD = 70

# Create face dataset directory if not exists
if not os.path.exists(FACE_DATASET):
    os.makedirs(FACE_DATASET)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Configuration
BUFFER_DISTANCE = 30
ALERT_COLOR = (0, 0, 255)
CAR_BOUNDARY_COLOR = (0, 255, 255)
PLATE_COLOR = (0, 255, 0)
PLATE_BOX_COLOR = (255, 0, 0)

def train_face_recognizer():
    global recognizer, label_ids
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_ids = {}
    current_id = 0

    if not os.path.exists(FACE_DATASET):
        return None

    for root, dirs, files in os.walk(FACE_DATASET):
        for file in files:
            if file.lower().endswith(("jpg", "png", "jpeg")):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (100, 100))
                faces.append(img)
                labels.append(label_ids[label])

    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        return label_ids
    return None

label_ids = train_face_recognizer()

def detect_number_plate(car_roi, car_position):
    gray = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.02,
        minNeighbors=5,
        minSize=(50, 20),
        maxSize=(200, 80)
    )
    
    plate_info = []
    for (x, y, w, h) in plates:
        # Convert plate coordinates to original image space
        abs_x = car_position[0] + x
        abs_y = car_position[1] + y
        
        # Extract plate region with padding
        plate_img = car_roi[max(0,y-15):min(y+h+15,car_roi.shape[0]), 
                          max(0,x-5):min(x+w-5,car_roi.shape[1])]
        
        # Preprocess for better OCR
        plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        plate_img = cv2.medianBlur(plate_img, 3)
        # cv2.imshow("Preprocessed Plate after median blur", plate_img)
        _, plate_img = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow("Preprocessed Plate after thresholding", plate_img)
        # Perform OCR
        text = pytesseract.image_to_string(plate_img, 
                                          config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        clean_text = ''.join([c for c in text.strip() if c.isalnum()])
        
        if len(clean_text) > 3:
            plate_info.append({
                'coordinates': (abs_x, abs_y, w, h),
                'text': clean_text
            })
            
    return plate_info

def recognize_face(face_roi):
    if label_ids is None or recognizer is None:
        return "Untrained", 100.0
    
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        label_id, confidence = recognizer.predict(gray)
        
        if confidence < FACE_RECOGNITION_THRESHOLD:
            for name, id in label_ids.items():
                if id == label_id:
                    return name, confidence
        return "Unknown", confidence
    except:
        return "Error", 100.0

def process_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img, conf=0.5, verbose=False)
    
    cars = []
    people = []
    intrusions = 0
    all_plates = []

    # First pass for object detection
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            if cls == 2:  # Car
                car_box = box.xyxy[0].cpu().numpy().astype(int)
                cars.append(car_box)
                
                # Extract car ROI
                x1, y1, x2, y2 = car_box
                car_roi = frame[y1:y2, x1:x2]
                
                if car_roi.size > 0:
                    # Detect number plates with absolute coordinates
                    plates = detect_number_plate(car_roi, (x1, y1))
                    all_plates.extend(plates)

            elif cls == 0:  # Person
                people.append(box.xyxy[0].cpu().numpy().astype(int))


    for plate in all_plates:
        x, y, w, h = plate['coordinates']
        # Draw plate bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), PLATE_BOX_COLOR, 2)
        # Draw plate text
        cv2.putText(img, plate['text'], (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLATE_COLOR, 2)
        
    # Car safety zones and intrusion detection
    car_boundaries = []
    for car in cars:
        x1, y1, x2, y2 = map(int, car)
        x1 = max(0, x1 - BUFFER_DISTANCE)
        y1 = max(0, y1 - BUFFER_DISTANCE)
        x2 = min(img.shape[1], x2 + BUFFER_DISTANCE)
        y2 = min(img.shape[0], y2 + BUFFER_DISTANCE)
        car_boundaries.append((x1, y1, x2, y2))
        
        # Draw original car box
        cv2.rectangle(img, (int(car[0]), int(car[1])), 
                     (int(car[2]), int(car[3])), (0, 255, 0), 2)

    # Intrusion check
    for person in people:
        px1, py1, px2, py2 = map(int, person)
        person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
        person_roi = frame[py1:py2, px1:px2]
        
        if person_roi.size == 0:
            continue
            
        # Detect faces
        gray_person = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_person,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (fx, fy, fw, fh) in faces:
            # Draw face rectangle
            cv2.rectangle(img, (px1+fx, py1+fy), 
                         (px1+fx+fw, py1+fy+fh), (255, 0, 0), 2)
            
            # Recognize face
            face_roi = person_roi[fy:fy+fh, fx:fx+fw]
            name, confidence = recognize_face(face_roi)
            
            # Display recognition result
            text = f"{name} ({confidence:.1f})"
            cv2.putText(img, text, (px1+fx, py1+fy-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        intrusion_detected = False
        for i, (cx1, cy1, cx2, cy2) in enumerate(car_boundaries):
            if (cx1 <= person_center[0] <= cx2 and 
                cy1 <= person_center[1] <= cy2):
                cv2.rectangle(img, (cx1, cy1), (cx2, cy2), ALERT_COLOR, 2)
                intrusion_detected = True
                intrusions += 1
                break
        
        color = ALERT_COLOR if intrusion_detected else (0, 255, 0)
        cv2.rectangle(img, (px1, py1), (px2, py2), color, 2)

    # Draw plates and texts
    for i, plate_text in enumerate(all_plates):
        cv2.putText(img, f'Plate: {plate_text}', (10, 120 + i*30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLATE_COLOR, 2)

    # Display counters
    cv2.putText(img, f'Cars: {len(cars)}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f'People: {len(people)}', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f'Intrusions: {intrusions}', (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, ALERT_COLOR, 2)
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_frame(frame)
    
    cv2.waitKey(1)
    cv2.imshow('Integrated Monitoring System', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
