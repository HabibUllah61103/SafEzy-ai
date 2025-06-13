from flask import Flask, Response, jsonify
import threading, time, socket, json, cv2, numpy as np
from ultralytics import YOLO
import pytesseract, face_recognition, psycopg2
import psycopg2.extras
from datetime import datetime
import logging
import os
import requests

# ————— Flask Setup —————
app = Flask(__name__)
monitoring_active = False
monitor_thread = None
latest_frame = None
frame_lock = threading.Lock()
log_lock = threading.Lock()

# ————— Tesseract & Models —————
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
model = YOLO('yolov8n.pt')
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ————— Config & Globals —————
BUFFER = 30
ALERT_COLOR = (0, 0, 255)
CAR_COLOR = (0, 255, 0)
PLATE_BOX = (255, 0, 0)
PLATE_TEXT_COLOR = (0, 255, 0)
FACE_COLOR = (255, 0, 0)
LOG_FILE = 'security_logs.txt'

# Configure logging
logging.basicConfig(level=logging.INFO)

# ————— DB & Socket Config —————
PRIMARY_DB = {
    'host': 'bbewj40tjy1rqjuplnen-postgresql.services.clever-cloud.com',
    'port': 50013,
    'user': 'udbrqsl8p6iactizkwkx',
    'password': 'Imtyr83b5Sj6fpKoXOjv',
    'dbname': 'bbewj40tjy1rqjuplnen'
}  # same as your original
SERVER_HOST, SERVER_PORT = 'localhost', 3000

# ————— Known Face Setup —————
known_encodings, known_names = [], []
try:
    img = face_recognition.load_image_file("./known_faces/owner.jpg")
    enc = face_recognition.face_encodings(img)[0]
    known_encodings.append(enc); known_names.append("Habib Ullah")
except Exception: pass

# ————— Helpers —————
def connect_db(cfg):
    try:
        return psycopg2.connect(**cfg)
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return None

def log_message(msg):
    timestamp = datetime.now().isoformat()
    with log_lock:
        with open(LOG_FILE, 'a') as f:
            f.write(f"{timestamp} - {msg}\n")
    # print(f"{timestamp} | {msg}")

def fetch_owner_and_vehicles(device_id=1):
    try:
        conn = connect_db(PRIMARY_DB)
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        cur.execute("SELECT owner_id FROM device WHERE id = %s", (device_id,))
        device = cur.fetchone()
        if not device:
            logging.error("Device ID not found")
            return None, None, None

        owner_id = device['owner_id']
        logging.info(f"Device {device_id} belongs to owner {owner_id}")

        cur.execute("""
            SELECT *
            FROM vehicle v
            WHERE v.owner_id = %s
        """, (owner_id,))
        vehicles = cur.fetchall()

        cur.execute("""
            SELECT profile_image_url
            FROM "user"
            WHERE id = %s
        """, (owner_id,))
        user = cur.fetchone()

        conn.close()
        print(owner_id, vehicles, user['profile_image_url'])
        return owner_id, vehicles, user['profile_image_url']
    except Exception as e:
        logging.error(f"DB Error: {e}")
        return None, None, None

def save_vehicle_images(vehicles, directory='authorized_vehicles'):
    os.makedirs(directory, exist_ok=True)
    plate_numbers = []

    for vehicle in vehicles:
        plate = vehicle['numberPlate']
        image_urls = vehicle['image']
        for url in image_urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                if url:
                    try:
                        img_data = requests.get(url).content
                        with open(os.path.join(directory, f"{plate}.jpg"), 'wb') as handler:
                            handler.write(img_data)
                        logging.info(f"Saved image for {plate}")
                    except Exception as e:
                        logging.warning(f"Failed to download image for {plate}: {e}")
                else:
                    logging.warning(f"No image URL for {plate}")
                # Proceed with saving image or processing
            except Exception as e:
                logging.warning(f"Failed to download image from {url}: {e}")
        plate_numbers.append(plate)


    return plate_numbers

def save_owner_image(url, filename="owner.jpg", directory='known_faces'):
    os.makedirs(directory, exist_ok=True)
    if url:
        try:
            img_data = requests.get(url).content
            with open(os.path.join(directory, filename), 'wb') as handler:
                handler.write(img_data)
            logging.info("Saved owner image")
        except Exception as e:
            logging.warning(f"Failed to download owner image: {e}")


def log_alert_db(plate, count):
    conn = connect_db(PRIMARY_DB)
    if not conn: return
    with conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("INSERT INTO alerts(timestamp, plate_number, intrusion_count) VALUES (%s,%s,%s)",
                    (datetime.now(), plate, count))
    conn.close()

def send_alert(plate, count):
    payload = {'plate_number': plate, 'intrusion_count': count, 'timestamp': str(datetime.now())}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((SERVER_HOST, SERVER_PORT))
            s.send(json.dumps(payload).encode())
    except Exception as e:
        log_message(f"Socket alert error: {e}")

# ————— Image Processing —————
def preprocess_plate(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2, fy=2)
    g = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4)).apply(g)
    g = cv2.bilateralFilter(g, 150,180,180)
    g = cv2.adaptiveThreshold(g,240,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,4)
    k = cv2.getStructuringElement(cv2.MORPH_OPEN, (4,4))
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, k)
    return cv2.filter2D(g, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))

def detect_plates(roi, offset):
    plates, results = [], plate_cascade.detectMultiScale(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),1.02,5, minSize=(50,20), maxSize=(200,80))
    for x,y,w,h in results:
        img = roi[y:y+h, x:x+w]
        text = pytesseract.image_to_string(preprocess_plate(img), config='--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        clean = ''.join(filter(str.isalnum, text))
        if 3 <= len(clean) <= 8:
            plates.append({'coords': (offset[0]+x, offset[1]+y,w,h), 'text': clean})
    return plates

def annotate_frame(frame):
    global latest_frame
    cars, people, intrusions, plates_bundle = [], [], 0, []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ✅ Face detection and recognition block starts here
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_names[matched_idx]

        cv2.rectangle(frame, (left, top), (right, bottom), FACE_COLOR, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FACE_COLOR, 2)
    # ✅ End of face recognition block

    for res in model.predict(rgb, conf=0.5, verbose=False):
        for b in res.boxes:
            cls,x1,y1,x2,y2 = int(b.cls), *b.xyxy[0].cpu().numpy().astype(int)
            if cls == 2:  # Car
                cars.append((x1,y1,x2,y2))
                plates_bundle += detect_plates(frame[y1:y2, x1:x2], (x1,y1))
            elif cls == 0:
                people.append((x1,y1,x2,y2))

    boundaries = [(x1-BUFFER, y1-BUFFER, x2+BUFFER, y2+BUFFER) for x1,y1,x2,y2 in cars]
    for px1,py1,px2,py2 in people:
        cx, cy = (px1+px2)//2, (py1+py2)//2
        intruded = any(x1<=cx<=x2 and y1<=cy<=y2 for x1,y1,x2,y2 in boundaries)
        if intruded: intrusions += 1
        color = ALERT_COLOR if intruded else CAR_COLOR
        cv2.rectangle(frame,(px1,py1),(px2,py2),color,2)

    for x1,y1,x2,y2 in cars:
        cv2.rectangle(frame,(x1,y1),(x2,y2),CAR_COLOR,2)
    for p in plates_bundle:
        x,y,w,h = p['coords']
        cv2.rectangle(frame,(x,y),(x+w,y+h),PLATE_BOX,1)
        cv2.putText(frame,p['text'],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,PLATE_TEXT_COLOR,1)

    cv2.putText(frame,f"Cars:{len(cars)} People:{len(people)} Intr:{intrusions}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    # last_frame safe update
    with frame_lock:
        ret, buf = cv2.imencode('.jpg', frame)
        if ret: latest_frame = buf.tobytes()

    # logging
    plates_str = ','.join(p['text'] for p in plates_bundle) or 'None'
    log_message(f"Frame processed – Cars:{len(cars)} People:{len(people)} Plates:{plates_str} Intrusions:{intrusions}")

    # alert
    if intrusions > 0:
        threading.Thread(target=send_alert, args=(plates_bundle[0]['text'] if plates_bundle else "UNKNOWN",intrusions)).start()
        threading.Thread(target=log_alert_db, args=(plates_bundle[0]['text'] if plates_bundle else "UNKNOWN",intrusions)).start()

def monitor_loop():
    global monitoring_active
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
    while monitoring_active:
        ret, frm = cap.read()
        if not ret: continue
        annotate_frame(frm)
    cap.release()

# ————— Flask Routes —————
@app.route('/')
def get_status():
    return jsonify({'monitor': monitoring_active})

@app.route('/start-monitor', methods=['POST'])
def start_monitor():
    global monitor_thread, monitoring_active
    if monitoring_active:
        return jsonify({'status':'Already running'}), 409
    monitoring_active = True
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    return jsonify({'status':'Monitor started'})

@app.route('/stop-monitor', methods=['POST'])
def stop_monitor():
    global monitoring_active
    monitoring_active = False
    return jsonify({'status':'Monitor stopped'})

@app.route('/video-feed')
def video_feed():
    def stream():
        while monitoring_active:
            with frame_lock:
                if latest_frame:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            time.sleep(0.05)
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ————— Run Server —————
if __name__ == '__main__':
    logging.info("Initializing system...")

    owner_id, vehicles, owner_img_url = fetch_owner_and_vehicles(device_id=1)
    if vehicles and owner_id:
        authorized_plates = save_vehicle_images(vehicles)
        save_owner_image(owner_img_url)

        # Persist in memory for access during stream
        app.config['AUTHORIZED_PLATES'] = set(authorized_plates)
        app.config['OWNER_ID'] = owner_id
    else:
        logging.error("Failed to initialize owner data.")

    app.run(host='0.0.0.0', port=5000, debug=False)

