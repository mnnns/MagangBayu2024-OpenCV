import cv2
import numpy as np

def detect_faces(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    return image

def run_face_detection():
    webcam = cv2.VideoCapture(0)
    while True:
        ret, frame = webcam.read()
        frame_with_faces = detect_faces(frame)
        cv2.imshow('Face Detection (press q to quit)', frame_with_faces)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_detection()