import cv2
import numpy as np
import mysql.connector
from datetime import datetime
import os
import sys
import time
from deepface import DeepFace

cv2.setNumThreads(2)  # Enable parallelism if supported

class EmotionDetectorHighAccuracy:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_labels = ['sad', 'angry', 'neutral', 'happy']
        self.db_connection = None
        self.setup_database()
        self.setup_emotion_model()

    def setup_database(self):
        try:
            db_config = {
                'host': 'localhost',
                'user': 'root',
                'password': 'pralakshy@16',
                'database': 'emotional_detection'
            }
            self.db_connection = mysql.connector.connect(**db_config)
            cursor = self.db_connection.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS emotion_detection")
            cursor.execute("USE emotion_detection")
            create_table_query = """
                                 CREATE TABLE IF NOT EXISTS emotion_records \
                                 ( \
                                     id INT AUTO_INCREMENT PRIMARY KEY, \
                                     timestamp DATETIME NOT NULL, \
                                     emotion VARCHAR(20) NOT NULL, \
                                     confidence FLOAT NOT NULL, \
                                     face_detected BOOLEAN NOT NULL, \
                                     image_path VARCHAR(255)
                                 )
                                 """
            cursor.execute(create_table_query)
            self.db_connection.commit()
            print("Database setup completed successfully!")
        except mysql.connector.Error as err:
            print(f"Database error: {err}")
            print("Please make sure MySQL is running and credentials are correct")
            sys.exit(1)

    def setup_emotion_model(self):
        try:
            dummy_img = np.zeros((48, 48, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False)
            print("DeepFace emotion detection model initialized successfully!")
        except Exception as e:
            print(f"Warning: DeepFace initialization failed: {e}")
            print("Falling back to basic emotion detection")

    def detect_emotion_deepface(self, face_roi):
        try:
            if len(face_roi.shape) == 3:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            result = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
            emotions = result[0]['emotion']
            emotion_mapping = {
                'sad': emotions.get('sad', 0),
                'angry': emotions.get('angry', 0),
                'neutral': emotions.get('neutral', 0),
                'happy': emotions.get('happy', 0)
            }
            detected_emotion = max(emotion_mapping, key=emotion_mapping.get)
            confidence = float(emotion_mapping[detected_emotion]) / 100.0  # Convert to [0,1]
            return detected_emotion, confidence
        except Exception as e:
            print(f"DeepFace error: {e}")
            # If DeepFace fails, fall back to basic neutral
            return 'neutral', 0.0

    def save_to_database(self, emotion, confidence, face_detected, image_path=None):
        try:
            cursor = self.db_connection.cursor()
            py_confidence = float(confidence)
            py_face_detected = int(bool(face_detected))
            values = (datetime.now(), emotion, py_confidence, py_face_detected, image_path)
            insert_query = """
                INSERT INTO emotion_records (timestamp, emotion, confidence, face_detected, image_path)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, values)
            self.db_connection.commit()
            print(f"Saved to database: {emotion} (confidence: {py_confidence:.2f})")
        except mysql.connector.Error as err:
            print(f"Database save error: {err}")

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            emotion, confidence = self.detect_emotion_deepface(face_roi)
            results.append({
                'rect': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'face_roi': face_roi
            })
        return results

    def run_emotion_detection(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        print("Emotion Detection (High Accuracy) Started!")
        print("Press 'q' to quit, 's' to save current frame")
        frame_count = 0
        save_interval = 30
        session_emotions = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0}
        session_confidences = []
        start_time = time.time()
        last_fps_print_time = start_time
        fps = 0.0
        fps_count = 0
        fps_timer = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                frame_to_show = frame.copy()
                results = self.process_frame(frame)
                for r in results:
                    x, y, w, h = r['rect']
                    emotion = r['emotion']
                    confidence = r['confidence']
                    session_confidences.append(confidence)
                    if emotion in session_emotions:
                        session_emotions[emotion] += 1
                    percent = int(confidence * 100)
                    label_text = f"{emotion} ({percent}%)"
                    # Draw bold-like
                    cv2.putText(frame_to_show, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                    cv2.putText(frame_to_show, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    cv2.rectangle(frame_to_show, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # FPS calculation
                fps_count += 1
                fps_elapsed = time.time() - fps_timer
                if fps_elapsed >= 1.0:
                    fps = fps_count / fps_elapsed
                    fps_count = 0
                    fps_timer = time.time()
                # Show session stats
                y_offset = 30
                cv2.putText(frame_to_show, "Session Stats:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
                for emotion_name, count in session_emotions.items():
                    if count > 0:
                        cv2.putText(frame_to_show, f"{emotion_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_offset += 20
                # Display FPS below stats
                cv2.putText(frame_to_show, f"FPS: {fps:.2f}", (10, y_offset+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 150, 255), 2)
                # Save each detected face to DB every save_interval frames
                frame_count += 1
                if frame_count % save_interval == 0:
                    for r in results:
                        self.save_to_database(r['emotion'], r['confidence'], True)
                # Show window with fps in the title
                cv2.imshow(f'Emotion Detection High Accuracy [FPS: {fps:.2f}]', frame_to_show)
                # Print FPS every ~3 seconds
                now = time.time()
                if now - last_fps_print_time > 3:
                    print(f"Current FPS: {fps:.2f}")
                    last_fps_print_time = now
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Saving and exiting...")
                    for r in results:
                        self.save_to_database(r['emotion'], r['confidence'], True)
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"emotion_frame_{timestamp}.jpg"
                    cv2.imwrite(image_path, frame_to_show)
                    for r in results:
                        self.save_to_database(r['emotion'], r['confidence'], True, image_path)
                    print(f"Frame saved as {image_path}")
                # Stop after 30 seconds
                if now - start_time >= 30:
                    print("30 seconds elapsed. Stopping and saving...")
                    for r in results:
                        self.save_to_database(r['emotion'], r['confidence'], True)
                    break
        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Saving...")
            for r in results:
                self.save_to_database(r['emotion'], r['confidence'], True)
        except Exception as e:
            print(f"An error occurred: {e}")
            for r in results:
                self.save_to_database(r['emotion'], r['confidence'], True)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.db_connection:
                self.get_emotion_statistics()
                self.db_connection.close()
            confs = session_confidences
            if confs:
                avg_conf = sum(confs) / len(confs)
                print(f"Average Confidence this session: {avg_conf*100:.1f} %")
            print("High-accuracy emotion detection stopped!")

            # Results popup window
            try:
                width, height = 600, 420
                results_img = np.ones((height, width, 3), dtype=np.uint8) * 255
                y = 40
                cv2.putText(results_img, 'Session Results (30s)', (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                y += 40
                cv2.putText(results_img, 'Emotion Counts:', (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                y += 30
                for emotion_name in ['happy', 'sad', 'angry', 'neutral']:
                    count = session_emotions.get(emotion_name, 0)
                    cv2.putText(results_img, f'{emotion_name}: {count}', (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
                    y += 28
                avg_conf = (sum(session_confidences) / len(session_confidences)) if session_confidences else 0.0
                y += 10
                cv2.putText(results_img, f'Average Confidence: {avg_conf*100:.1f}%', (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 100, 0), 2)
                y += 35
                cv2.putText(results_img, 'Press any key to close', (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 200), 2)
                cv2.namedWindow('Session Results', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Session Results', results_img)
                cv2.waitKey(0)
            finally:
                cv2.destroyAllWindows()

    def get_emotion_statistics(self):
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM emotion_records
                WHERE face_detected = 1
                GROUP BY emotion
                ORDER BY count DESC
            """)
            results = cursor.fetchall()
            print("\n=== Emotion Detection Statistics ===")
            print("Emotion\t\tCount\tAvg Confidence")
            print("-" * 40)
            for emotion, count, avg_conf in results:
                print(f"{emotion:<12}\t{count}\t{avg_conf:.2f}")
            cursor.execute("SELECT COUNT(*) FROM emotion_records")
            total_records = cursor.fetchone()[0]
            print(f"\nTotal records: {total_records}")
        except mysql.connector.Error as err:
            print(f"Database statistics error: {err}")

def main():
    print("=== High-Accuracy Emotion Detection System ===")
    print("This program detects 4 emotions (Happy, Sad, Angry, Neutral) from webcam (full frame accuracy, DeepFace)")
    print("and stores results in MySQL database")
    print("\nRequirements:")
    print("1. MySQL server running")
    print("2. Webcam connected")
    print("3. Required Python packages installed")

    try:
        import cv2
        import mysql.connector
        import numpy as np
        from deepface import DeepFace
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages using: pip install -r requirements.txt")
        return

    detector = EmotionDetectorHighAccuracy()
    detector.run_emotion_detection()

if __name__ == "__main__":
    main()


