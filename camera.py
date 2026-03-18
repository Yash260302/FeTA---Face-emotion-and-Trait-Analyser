# camera.py
import cv2

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
_face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_faces(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    """
    frame: BGR image (numpy array)
    returns: list of (x, y, w, h)
    """
    if _face_cascade.empty():
        raise IOError("Failed to load Haar cascade at " + cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize
    )
    return faces
