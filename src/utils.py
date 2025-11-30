import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def load_labels(models_dir='../models'):
    labels_path = os.path.join(models_dir, 'labels.json')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json not found in {models_dir}")
    with open(labels_path, 'r') as f:
        class_map = json.load(f)
    # class_map: index -> label
    labels = [class_map[str(i)] if isinstance(list(class_map.keys())[0], str) else class_map[i] for i in range(len(class_map))]
    # ensure order by index
    labels = [class_map[str(i)] if str(i) in class_map else class_map[i] for i in range(len(class_map))]
    return labels


def preprocess_image_bgr(img_bgr, img_size=64):
    # img_bgr is an image loaded by cv2 (BGR)
    import cv2
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_arr = img_to_array(img_resized) / 255.0
    return np.expand_dims(img_arr, axis=0)


def load_trained_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = load_model(model_path)
    return model


def predict_from_frame(model, frame_bgr, labels, img_size=64):
    x = preprocess_image_bgr(frame_bgr, img_size=img_size)
    preds = model.predict(x)
    idx = int(np.argmax(preds))
    prob = float(np.max(preds))
    label = labels[idx]
    return label, prob