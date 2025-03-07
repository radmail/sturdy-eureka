import cv2
import numpy as np
from keras.models import load_model
from playsound import playsound
import threading

def start_alarm():
    """Play the alarm sound"""
    playsound('data/alarm.mp3')

def detect_eyes(frame):
    """Detect eyes in the frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    return eyes

def process_frame(frame):
    """Process the frame and detect drowsiness"""
    global count, alarm_on

    if len(frame.shape) != 3:
        print("Input image is not a color image!")
        return

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect faces and eyes
    faces = face_cascade.detectMultiScale(hsv, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_hsv = hsv[y:y+h, x:x+w]
        roi_bgr = frame[y:y+h, x:x+w]
        eyes = detect_eyes(roi_bgr)

        # If eyes are detected, process them
        if len(eyes) > 1:
            for (x1, y1, w1, h1) in eyes:
                cv2.rectangle(roi_bgr, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
                eye = roi_bgr[y1:y1+h1, x1:x1+w1]
                eye = cv2.resize(eye, (145, 145))
                eye = eye.astype('float') / 255.0
                eye = np.expand_dims(eye, axis=0)

                # Make a prediction
                pred = model.predict(eye)
                status = np.argmax(pred)

                # If the eyes are closed, start counting
                if status == 2:
                    count += 1
                    cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

                    # If eyes are closed for 10 consecutive frames, start the alarm
                    if count >= 10:
                        cv2.putText(frame, "Drowsiness Alert!!!", (100, height-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        if not alarm_on:
                            alarm_on = True
                            # Start the alarm sound in a new thread
                            t = threading.Thread(target=start_alarm)
                            t.daemon = True
                            t.start()

                else:
                    count = 0
                    alarm_on = False

# Load the models and cascades
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye.xml")
model = load_model("drowiness_new7.h5")

# Initialize the global variables
count = 0
alarm_on = False

# Start the alarm sound thread
alarm_thread = threading.Thread(target=start_alarm)
alarm_thread.daemon = True
alarm_thread.start()

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    height = frame.shape[0]

    # Process the frame
    process_frame(frame)

    # Display the frame
    cv2.imshow("Drowsiness Detector", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()