import sys
import time
from datetime import datetime

import cv2
import numpy as np
from deepface import DeepFace

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QFrame,
)


class WebcamRunner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Detection - Studio UI")
        self.resize(1200, 800)

        # Global dark theme: dark indigo background, unified dark grey borders, bold Inter-like font
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1a1f3b; /* dark indigo */
                color: #e6e6e6;
                font-family: 'Inter', 'SF Pro Text', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                font-weight: 600; /* bold-like */
            }
            QSplitter::handle { background: #2a2f49; width: 6px; }

            QPushButton {
                background-color: #222844;
                border: 3px solid #3f434a; /* unified dark grey */
                padding: 8px 14px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #283058; }
            QPushButton:disabled { color: #9a9a9a; border-color: #33363c; background-color: #1b2039; }

            QLineEdit {
                background-color: #192042;
                border: 3px solid #3f434a; /* unified dark grey */
                padding: 8px 10px;
                border-radius: 8px;
                color: #f0f0f0;
            }

            QTextEdit {
                background-color: #12162d;
                border: 3px solid #3f434a; /* unified dark grey */
                border-radius: 12px;
            }

            #VideoFrame { 
                border: 4px solid #3f434a; /* unified dark grey */
                border-radius: 14px; 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #171c38, stop:1 #202756);
            }
            #ChatFrame {
                border: 4px solid #3f434a; /* unified dark grey */
                border-radius: 14px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #161b36, stop:1 #1f2650);
            }
            #TerminalFrame {
                border: 4px solid #3f434a; /* unified dark grey */
                border-radius: 14px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #151938, stop:1 #1c224a);
            }

            #VideoLabel { background: transparent; color: #f0f0f0; }
            #ChatHistory { background-color: rgba(0,0,0,0); border: none; }
            #TerminalEdit { background-color: rgba(0,0,0,0); border: none; }
            """
        )

        # Layout: [ Main (video) | Side (chat) ] over Bottom (terminal)
        root_layout = QVBoxLayout(self)

        # --- Top Split: Video (left) and Chat (right)
        top_split = QSplitter(Qt.Horizontal)
        top_split.setChildrenCollapsible(False)

        # Left: Video panel in a framed container
        self.video_label = QLabel()
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        video_container = QFrame()
        video_container.setObjectName("VideoFrame")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(8, 8, 8, 8)
        video_layout.addWidget(self.video_label)

        # Right: Chat panel in a framed container
        chat_container = QFrame()
        chat_container.setObjectName("ChatFrame")
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(8, 8, 8, 8)
        self.chat_history = QTextEdit()
        self.chat_history.setObjectName("ChatHistory")
        self.chat_history.setReadOnly(True)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Add a comment...")
        self.chat_send = QPushButton("Send")
        self.chat_send.clicked.connect(self._on_send_chat)
        chat_layout.addWidget(self.chat_history)
        chat_bottom = QHBoxLayout()
        chat_bottom.addWidget(self.chat_input)
        chat_bottom.addWidget(self.chat_send)
        chat_layout.addLayout(chat_bottom)

        top_split.addWidget(video_container)
        top_split.addWidget(chat_container)
        top_split.setSizes([900, 300])

        # --- Bottom: Terminal panel in a framed container
        terminal_container = QFrame()
        terminal_container.setObjectName("TerminalFrame")
        terminal_layout = QVBoxLayout(terminal_container)
        terminal_layout.setContentsMargins(8, 8, 8, 8)
        self.terminal = QTextEdit()
        self.terminal.setObjectName("TerminalEdit")
        self.terminal.setReadOnly(True)
        terminal_layout.addWidget(self.terminal)

        root_split = QSplitter(Qt.Vertical)
        root_split.setChildrenCollapsible(False)
        root_split.addWidget(top_split)
        root_split.addWidget(terminal_container)
        root_split.setSizes([600, 200])

        root_layout.addWidget(root_split)

        # Controls row (Start/Stop)
        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start (30s)")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_run)
        self.stop_btn.clicked.connect(self.stop_run)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)
        root_layout.addLayout(controls)

        # Video/processing state
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.run_started_at = None
        self.session_emotions = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0}
        self.session_confidences = []
        self.last_frame_time = time.time()
        self.fps_count = 0
        self.fps = 0.0
        self.fps_timer = time.time()

        # Pre-load deepface
        self._log("Initializing DeepFace model...")
        try:
            dummy_img = np.zeros((48, 48, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False)
            self._log("DeepFace ready.")
        except Exception as e:
            self._log(f"DeepFace init warning: {e}")

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.terminal.append(f"[{ts}] {msg}")

    def _on_send_chat(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self.chat_history.append(f"[{ts}] You: {text}")
        self.chat_input.clear()

    def start_run(self):
        if self.cap is not None:
            self._log("Already running.")
            return
        self._log("Starting webcam session (30s)...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            self._log("Error: Could not open webcam")
            self.cap.release()
            self.cap = None
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.run_started_at = time.time()
        self.session_emotions = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0}
        self.session_confidences = []
        self.fps = 0.0
        self.fps_count = 0
        self.fps_timer = time.time()
        self.timer.start(0)  # as fast as possible

    def stop_run(self):
        if self.cap is None:
            return
        self.timer.stop()
        self.cap.release()
        self.cap = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._show_results()
        self._log("Session stopped.")

    def _on_tick(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self._log("Error: Could not read frame")
            self.stop_run()
            return

        # Process frame at full resolution for higher accuracy
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue
            try:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                result = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
                emotions = result[0]['emotion']
                emotion_mapping = {
                    'sad': emotions.get('sad', 0),
                    'angry': emotions.get('angry', 0),
                    'neutral': emotions.get('neutral', 0),
                    'happy': emotions.get('happy', 0)
                }
                detected_emotion = max(emotion_mapping, key=emotion_mapping.get)
                confidence = float(emotion_mapping[detected_emotion]) / 100.0
            except Exception as e:
                self._log(f"DeepFace error: {e}")
                detected_emotion = 'neutral'
                confidence = 0.0

            if detected_emotion in self.session_emotions:
                self.session_emotions[detected_emotion] += 1
            self.session_confidences.append(confidence)

            # Draw overlays
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (90, 140, 250), 3)
            percent = int(confidence * 100)
            label_text = f"{detected_emotion} ({percent}%)"
            cv2.putText(display_frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(display_frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 255), 2)

        # FPS update
        self.fps_count += 1
        elapsed = time.time() - self.fps_timer
        if elapsed >= 1.0:
            self.fps = self.fps_count / elapsed
            self.fps_count = 0
            self.fps_timer = time.time()

        # Session stats overlay
        y_offset = 30
        cv2.putText(display_frame, "Session Stats:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2)
        y_offset += 25
        for emotion_name, count in self.session_emotions.items():
            if count > 0:
                cv2.putText(display_frame, f"{emotion_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1)
                y_offset += 22
        cv2.putText(display_frame, f"FPS: {self.fps:.2f}", (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 220, 140), 2)

        # Render to QLabel
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        # Time limit 30s
        if time.time() - self.run_started_at >= 30:
            self._log("30 seconds elapsed. Finishing run...")
            self.stop_run()

    def _show_results(self):
        # Print results to terminal panel
        self._log("=== Session Results ===")
        for emotion_name in ['happy', 'sad', 'angry', 'neutral']:
            count = self.session_emotions.get(emotion_name, 0)
            self._log(f"{emotion_name}: {count}")
        avg_conf = (sum(self.session_confidences) / len(self.session_confidences)) if self.session_confidences else 0.0
        self._log(f"Average Confidence: {avg_conf*100:.1f}%")


def main():
    app = QApplication(sys.argv)
    w = WebcamRunner()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
