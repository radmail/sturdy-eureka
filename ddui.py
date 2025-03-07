import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from threading import Thread
import playsound


class DrowsinessDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Detection")

        self.video_label = QLabel()
        self.start_button = QPushButton("Start Detection")
        self.stop_button = QPushButton("Stop Detection")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)

        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_drowsiness)

        self.classes = ['Closed', 'Open']
        self.face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
        self.left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
        self.right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
        self.model = load_model("drowiness_new7.h5")
        self.count = 0
        self.alarm_on = False
        self.alarm_sound = "data/alarm.mp3"
        self.status1 = ''
        self.status2 = ''

    @pyqtSlot()
    def start_detection(self):
        self.capture = cv2.VideoCapture(0)
        self.timer.start(30)  # 30 milliseconds between each frame

    @pyqtSlot()
    def stop_detection(self):
        if self.capture:
            self.capture.release()
            self.timer.stop()
            self.video_label.clear()

    def detect_drowsiness(self):
        ret, frame = self.capture.read()
        if ret:
            # Process frame
            height, width, _ = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                left_eye = self.left_eye_cascade.detectMultiScale(roi_gray)
                right_eye = self.right_eye_cascade.detectMultiScale(roi_gray)
                for (x1, y1, w1, h1) in left_eye:
                    cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
                    eye1 = roi_color[y1:y1 + h1, x1:x1 + w1]
                    eye1 = cv2.resize(eye1, (145, 145))
                    eye1 = eye1.astype('float') / 255.0
                    eye1 = img_to_array(eye1)
                    eye1 = np.expand_dims(eye1, axis=0)
                    pred1 = self.model.predict(eye1)
                    self.status1 = np.argmax(pred1)
                    break

                for (x2, y2, w2, h2) in right_eye:
                    cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
                    eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
                    eye2 = cv2.resize(eye2, (145, 145))
                    eye2 = eye2.astype('float') / 255.0
                    eye2 = img_to_array(eye2)
                    eye2 = np.expand_dims(eye2, axis=0)
                    pred2 = self.model.predict(eye2)
                    self.status2 = np.argmax(pred2)
                    break

                # If the eyes are closed, start counting
                if self.status1 == self.classes.index('Closed') and self.status2 == self.classes.index('Closed'):
                    self.count += 1
                    cv2.putText(frame, "Eyes Closed, Frame count: " + str(self.count), (10, 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    # if eyes are closed for 10 consecutive frames, start the alarm
                    if self.count >= 10:
                        cv2.putText(frame, "Drowsiness Alert!!!", (100, height - 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 0, 255), 2)
                        if not self.alarm_on:
                            self.alarm_on = True
                            # play the alarm sound
                            playsound.playsound(self.alarm_sound, True)
                else:
                    cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                    self.count = 0
                    self.alarm_on = False

            # Display the frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            q_img = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap)
            self.video_label.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessDetector()
    window.show()
    sys.exit(app.exec_())
