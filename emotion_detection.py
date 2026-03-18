
import os
import numpy as np
import cv2

# safe import of keras load_model
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

DEFAULT_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

class EmotionDetector:
    def __init__(self, model_path="models/best_emotion_model.h5", target_size=(48,48), emotion_labels=None):
        """
        model_path: path to a keras model. Model should accept either (H,W,1) grayscale or (H,W,3) RGB.
        target_size: (H,W) expected by model (default 48x48 common for FER).
        emotion_labels: list of class names in model output order.
        """
        self.model_path = model_path
        self.target_size = tuple(target_size)
        self.emotion_labels = emotion_labels if emotion_labels is not None else DEFAULT_EMOTIONS
        self.model = None
        self._expects_channels = 1
        self._noop_reason = None
        self._load_model()

    def _load_model(self):
        if load_model is None:
            self.model = None
            self._noop_reason = "tensorflow/keras not available"
            return

        if not os.path.exists(self.model_path):
            self.model = None
            self._noop_reason = f"Model file not found at '{self.model_path}'"
            return

        try:
            self.model = load_model(self.model_path, compile=False)
            # inspect input shape if available
            try:
                in_shape = self.model.input_shape  # e.g. (None, 48, 48, 1) or (None,48,48,3)
                if in_shape is not None and len(in_shape) == 4:
                    self._expects_channels = int(in_shape[-1]) if in_shape[-1] is not None else 1
                else:
                    self._expects_channels = 1
            except Exception:
                self._expects_channels = 1
            self._noop_reason = None
        except Exception as e:
            self.model = None
            self._noop_reason = f"Failed to load model: {e}"

    def is_ready(self):
        return self.model is not None

    def _preprocess_face(self, face_bgr):
        """
        face_bgr: cropped face image in BGR (numpy array)
        returns: preprocessed array (H,W,C) normalized to 0..1 or None on failure
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        h, w = self.target_size
        try:
            face_resized = cv2.resize(face_bgr, (w, h))
        except Exception:
            return None

        if getattr(self, "_expects_channels", 1) == 1:
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            arr = gray.astype("float32") / 255.0
            arr = np.expand_dims(arr, axis=-1)  # (H,W,1)
        else:
            rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            arr = rgb.astype("float32") / 255.0   # (H,W,3)
        return arr

    def _preprocess_batch(self, face_list):
        processed = []
        valid_indices = []
        for i, f in enumerate(face_list):
            p = self._preprocess_face(f)
            if p is None:
                continue
            processed.append(p)
            valid_indices.append(i)
        if not processed:
            return None, []
        return np.stack(processed, axis=0), valid_indices

    def predict_emotions_for_faces(self, frame_or_faces, faces=None):
        """
        Flexible API:
        - If called with (frame, faces) where faces is iterable of (x,y,w,h), it will crop and predict.
        - If called with (list_of_face_bgr_images, None) it will predict for those crops.
        Returns: (labels_list, confidences_list) aligned with the input faces order.
        If detector not ready returns ("Unknown", 0.0) for each face.
        """
        # two calling patterns supported
        if faces is None and isinstance(frame_or_faces, (list, tuple, np.ndarray)) and frame_or_faces and frame_or_faces[0].ndim >= 2:
            # called as predict_emotions_for_faces(list_of_bgr_crops)
            face_crops = list(frame_or_faces)
            boxes = None
        else:
            # called as predict_emotions_for_faces(frame, faces)
            if faces is None:
                return ([], [])
            frame = frame_or_faces
            boxes = list(faces)
            face_crops = []
            for (x, y, w, h) in boxes:
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + h))
                crop = frame[y1:y2, x1:x2]
                face_crops.append(crop if crop.size != 0 else None)

        if not self.is_ready():
            # return Unknown placeholders
            n = len(face_crops)
            return (["Unknown"] * n, [0.0] * n)

        batch, valid_indices = self._preprocess_batch(face_crops)
        if batch is None:
            # no valid crops
            return (["Unknown"] * len(face_crops), [0.0] * len(face_crops))

        try:
            preds = self.model.predict(batch, verbose=0)
        except Exception:
            return (["Unknown"] * len(face_crops), [0.0] * len(face_crops))

        # preds could be (N,C) array
        preds_arr = np.asarray(preds)
        if preds_arr.ndim == 1:
            # single value per sample? treat as Unknown
            labels = ["Unknown"] * len(face_crops)
            confs = [0.0] * len(face_crops)
            return labels, confs

        # if preds is list/tuple (some keras models), take first element if necessary
        if isinstance(preds, list) or isinstance(preds, tuple):
            preds_arr = np.asarray(preds[0])

        # interpret as probabilities across classes
        probs = preds_arr
        indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1).tolist()
        labels_out = []
        confs_out = []

        # build outputs aligned with original face_crops (some indices may be missing if invalid)
        labels_full = ["Unknown"] * len(face_crops)
        confs_full = [0.0] * len(face_crops)
        for out_i, orig_i in enumerate(valid_indices):
            idx = indices[out_i]
            label = self.emotion_labels[idx] if 0 <= idx < len(self.emotion_labels) else "Unknown"
            labels_full[orig_i] = label
            confs_full[orig_i] = float(confidences[out_i])

        return labels_full, confs_full

# ---- Singleton wrapper and convenience functions ----
_global_detector = EmotionDetector(model_path="models/best_emotion_model.h5")

def predict_emotion(face_bgr):
    """
    Accepts a single face crop (BGR numpy array) and returns a single label string.
    """
    if face_bgr is None or face_bgr.size == 0:
        return "Unknown"
    labels, confs = _global_detector.predict_emotions_for_faces([face_bgr])
    return labels[0] if labels else "Unknown"

def predict_emotions_for_faces(frame, faces):
    """
    Accepts a full frame and faces iterable (x,y,w,h), returns (labels, confidences).
    """
    return _global_detector.predict_emotions_for_faces(frame, faces)
