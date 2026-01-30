import cv2
import numpy as np
import mysql.connector
from datetime import datetime
import os
import sys
import time
from deepface import DeepFace

# Optimize OpenCV threads: modern CPUs like M4 are multi-core
cv2.setNumThreads(2)


class EmotionDetector:
    def __init__(self):
        # Initialize face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize emotion labels (only 4 emotions)
        self.emotion_labels = ['sad', 'angry', 'neutral', 'happy']

        # Initialize database connection
        self.db_connection = None
        self.setup_database()

        # Load emotion detection model (using a simple approach with OpenCV)
        # For better accuracy, you can use pre-trained models like FER2013 or other deep learning models
        self.setup_emotion_model()

    def setup_database(self):
        """Setup MySQL database connection and create tables if they don't exist"""
        try:
            # Database configuration - update these with your MySQL credentials
            db_config = {
                'host': 'localhost',
                'user': 'root',  # Change to your MySQL username
                'password': 'pralakshy@16',  # Change to your MySQL password
                'database': 'emotional_detection'
            }

            # Connect to MySQL server
            self.db_connection = mysql.connector.connect(**db_config)
            cursor = self.db_connection.cursor()

            # Create database if it doesn't exist
            cursor.execute("CREATE DATABASE IF NOT EXISTS emotion_detection")
            cursor.execute("USE emotion_detection")

            # Create emotions table if it doesn't exist
            create_table_query = """
                                 CREATE TABLE IF NOT EXISTS emotion_records \
                                 ( \
                                     id \
                                     INT \
                                     AUTO_INCREMENT \
                                     PRIMARY \
                                     KEY, \
                                     timestamp \
                                     DATETIME \
                                     NOT \
                                     NULL, \
                                     emotion \
                                     VARCHAR \
                                 ( \
                                     20 \
                                 ) NOT NULL,
                                     confidence FLOAT NOT NULL,
                                     face_detected BOOLEAN NOT NULL,
                                     image_path VARCHAR \
                                 ( \
                                     255 \
                                 )
                                     ) \
                                 """
            cursor.execute(create_table_query)
            self.db_connection.commit()
            print("Database setup completed successfully!")

        except mysql.connector.Error as err:
            print(f"Database error: {err}")
            print("Please make sure MySQL is running and credentials are correct")
            sys.exit(1)

    def setup_emotion_model(self):
        """Setup DeepFace emotion detection model"""
        try:
            # Test DeepFace with a dummy image to initialize the model
            dummy_img = np.zeros((48, 48, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False)
            print("DeepFace emotion detection model initialized successfully!")
        except Exception as e:
            print(f"Warning: DeepFace initialization failed: {e}")
            print("Falling back to basic emotion detection")

    def detect_emotion_deepface(self, face_roi):
        """
        Emotion detection using DeepFace library
        Returns emotion and confidence for the 4 target emotions
        """
        try:
            # Convert BGR to RGB for DeepFace
            if len(face_roi.shape) == 3:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

            # Analyze emotion using DeepFace
            result = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)

            # Extract emotion predictions
            emotions = result[0]['emotion']

            # Map DeepFace emotions to our 4 target emotions
            emotion_mapping = {
                'sad': emotions.get('sad', 0),
                'angry': emotions.get('angry', 0),
                'neutral': emotions.get('neutral', 0),
                'happy': emotions.get('happy', 0)
            }

            # Find the emotion with highest confidence
            detected_emotion = max(emotion_mapping, key=emotion_mapping.get)
            confidence = emotion_mapping[detected_emotion] / 100.0  # Convert percentage to decimal

            return detected_emotion, confidence

        except Exception as e:
            print(f"DeepFace error: {e}")
            # Fallback to basic detection
            return self.detect_emotion_fallback(face_roi)

    def detect_emotion_fallback(self, face_roi):
        """
        Fallback emotion detection when DeepFace fails
        Simple heuristic-based detection for 4 emotions
        """
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_roi

        # Resize face to standard size
        face_resized = cv2.resize(gray_face, (48, 48))
        height, width = face_resized.shape

        # Define facial regions
        eye_region = face_resized[:height // 3, :]
        mouth_region = face_resized[2 * height // 3:, :]

        # Calculate brightness
        eye_brightness = np.mean(eye_region)
        mouth_brightness = np.mean(mouth_region)

        # Calculate gradients for edge detection
        grad_x = cv2.Sobel(face_resized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(face_resized, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_density = np.mean(gradient_magnitude)

        # Simple rules for 4 emotions
        if mouth_brightness > eye_brightness + 15:
            emotion = 'happy'
            confidence = 0.7
        elif mouth_brightness < eye_brightness - 10 and edge_density > 30:
            emotion = 'angry'
            confidence = 0.6
        elif mouth_brightness < eye_brightness - 5:
            emotion = 'sad'
            confidence = 0.5
        else:
            emotion = 'neutral'
            confidence = 0.4

        return emotion, confidence

    def save_to_database(self, emotion, confidence, face_detected, image_path=None):
        """Save emotion detection result to MySQL database"""
        try:
            cursor = self.db_connection.cursor()
            insert_query = """
                           INSERT INTO emotion_records (timestamp, emotion, confidence, face_detected, image_path)
                           VALUES (%s, %s, %s, %s, %s) \
                           """

            # Ensure values are Python native types
            py_confidence = float(confidence)
            py_face_detected = int(bool(face_detected))

            values = (datetime.now(), emotion, py_confidence, py_face_detected, image_path)
            cursor.execute(insert_query, values)
            self.db_connection.commit()
            print(f"Saved to database: {emotion} (confidence: {py_confidence:.2f})")
        except mysql.connector.Error as err:
            print(f"Database save error: {err}")

    def get_emotion_statistics(self):
        """Get emotion detection statistics from database"""
        try:
            cursor = self.db_connection.cursor()

            # Get emotion counts
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

            # Get total records
            cursor.execute("SELECT COUNT(*) FROM emotion_records")
            total_records = cursor.fetchone()[0]
            print(f"\nTotal records: {total_records}")

        except mysql.connector.Error as err:
            print(f"Database statistics error: {err}")

    def process_frame(self, frame, session_emotions=None):
        """Process a single frame for emotion detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        emotion = "No Face"
        confidence = 0.0
        face_detected = False

        for (x, y, w, h) in faces:
            face_detected = True

            # Extract face ROI
            face_roi = gray[y:y + h, x:x + w]

            # Detect emotion using DeepFace
            emotion, confidence = self.detect_emotion_deepface(face_roi)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw emotion text with percentage in bold-like style
            percent = int(confidence * 100)
            label_text = f"{emotion} ({percent}%)"
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Display session statistics on frame
        if session_emotions:
            y_offset = 30
            cv2.putText(frame, "Session Stats:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25

            for emotion_name, count in session_emotions.items():
                if count > 0:
                    cv2.putText(frame, f"{emotion_name}: {count}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 20

        return frame, emotion, confidence, face_detected

    def run_emotion_detection(self):
        """Main function to run emotion detection from webcam"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        # Try to set higher FPS if hardware allows
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Emotion Detection Started!")
        print("Press 'q' to quit, 's' to save current frame")

        frame_count = 0
        save_interval = 30  # Save every 30 frames (about 1 second at 30fps)
        last_emotion = "No Face"
        last_confidence = 0.0
        last_face_detected = False
        session_emotions = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0}
        start_time = time.time()
        last_fps_print_time = start_time
        fps = 0.0
        fps_count = 0
        fps_timer = time.time()
        session_confidences = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Resize for detection (faster), but keep original for display
                display_frame = frame.copy()
                small_frame = cv2.resize(frame, (320, 240))
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 240

                # Convert once to grayscale for reuse
                gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray_small, 1.1, 4)

                emotion = "No Face"
                confidence = 0.0
                face_detected = False

                for (x, y, w, h) in faces:
                    # Scale coordinates back to original frame size
                    x_big, y_big = int(x * scale_x), int(y * scale_y)
                    w_big, h_big = int(w * scale_x), int(h * scale_y)
                    face_detected = True
                    face_roi = frame[y_big:y_big + h_big, x_big:x_big + w_big]
                    if face_roi.size == 0:
                        continue
                    emotion, confidence = self.detect_emotion_deepface(face_roi)
                    cv2.rectangle(display_frame, (x_big, y_big), (x_big + w_big, y_big + h_big), (255, 0, 0), 2)
                    percent = int(confidence * 100)
                    label_text = f"{emotion} ({percent}%)"
                    cv2.putText(display_frame, label_text, (x_big, y_big - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                    cv2.putText(display_frame, label_text, (x_big, y_big - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    break  # Just one face for performance

                # Track stats
                last_emotion = emotion
                last_confidence = confidence
                last_face_detected = face_detected
                if emotion in session_emotions:
                    session_emotions[emotion] += 1
                if face_detected:
                    session_confidences.append(float(confidence))

                # Overlay session statistics
                y_offset = 30
                cv2.putText(display_frame, "Session Stats:", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
                for emotion_name, count in session_emotions.items():
                    if count > 0:
                        cv2.putText(display_frame, f"{emotion_name}: {count}", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_offset += 20
                # Display FPS below stats
                cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, y_offset+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 150, 255), 2)
                # Save to database periodically
                frame_count += 1
                if frame_count % save_interval == 0:
                    self.save_to_database(emotion, confidence, face_detected)

                # Show window with fps in the title
                cv2.imshow(f'Emotion Detection [FPS: {fps:.2f}]', display_frame)

                # FPS display (print every ~3s)
                now = time.time()
                if now - last_fps_print_time > 3:
                    print(f"Current FPS: {fps:.2f}")
                    last_fps_print_time = now

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Saving final emotion data before exit...")
                    self.save_to_database(last_emotion, last_confidence, last_face_detected)
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"emotion_frame_{timestamp}.jpg"
                    cv2.imwrite(image_path, display_frame)
                    self.save_to_database(emotion, confidence, face_detected, image_path)
                    print(f"Frame saved as {image_path}")

                # Stop automatically after 30 seconds
                if now - start_time >= 30:
                    print("30 seconds elapsed. Saving final emotion data and stopping...")
                    self.save_to_database(last_emotion, last_confidence, last_face_detected)
                    break

        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Saving final data...")
            self.save_to_database(last_emotion, last_confidence, last_face_detected)

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Saving final emotion data...")
            self.save_to_database(last_emotion, last_confidence, last_face_detected)

        finally:
            # Cleanup - ensure this always runs
            cap.release()
            cv2.destroyAllWindows()

            # Show statistics before closing database
            if self.db_connection:
                self.get_emotion_statistics()
                self.db_connection.close()
            print("Emotion detection stopped!")

            # Results popup window
            try:
                width, height = 600, 400
                results_img = np.ones((height, width, 3), dtype=np.uint8) * 255
                y = 40
                cv2.putText(results_img, 'Session Results (30s)', (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                y += 40
                # Emotion counts
                cv2.putText(results_img, 'Emotion Counts:', (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                y += 30
                for emotion_name in ['happy', 'sad', 'angry', 'neutral']:
                    count = session_emotions.get(emotion_name, 0)
                    cv2.putText(results_img, f'{emotion_name}: {count}', (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
                    y += 28
                # Average confidence
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


def main():
    """Main function"""
    print("=== Emotion Detection System ===")
    print("This program detects 4 emotions (Happy, Sad, Angry, Neutral) from webcam using DeepFace")
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

    # Create and run emotion detector
    detector = EmotionDetector()
    detector.run_emotion_detection()


if __name__ == "__main__":
    main()