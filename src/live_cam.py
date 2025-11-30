import cv2
import time
from utils import load_trained_model, load_labels, preprocess_image_bgr

MODEL_PATH = 'models/asl_model_latest.h5'


def main():
    model = load_trained_model(MODEL_PATH)
    labels = load_labels(models_dir='models')

    cap = cv2.VideoCapture(0)
    time.sleep(1.0)

    if not cap.isOpened():
        print('Cannot open camera')
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            label, prob = predict(frame, model, labels)
            text = f'{label} {prob*100:.1f}%'
        except Exception as e:
            text = '---'

        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow('ASL - Simple Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict(frame, model, labels):
    label, prob = None, 0.0
    label, prob = None, 0.0
    from src.utils import predict_from_frame
    label, prob = predict_from_frame(model, frame, labels, img_size=64)
    return label, prob

if __name__ == '__main__':
    main()