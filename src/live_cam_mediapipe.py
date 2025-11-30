import cv2
import mediapipe as mp
import time
import numpy as np
from utils import load_trained_model, load_labels, preprocess_image_bgr

MODEL_PATH = 'models/asl_model_latest.h5'
PAD = 80  # pixels of padding around hand bbox


def main():
    model = load_trained_model(MODEL_PATH)
    labels = load_labels(models_dir='models')

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Cannot open camera')
        return

    # simple smoothing
    prev_label = None
    label_buffer = []
    buffer_len = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        display_text = '---'

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            x_coords = [int(p.x * w) for p in lm.landmark]
            y_coords = [int(p.y * h) for p in lm.landmark]
            x_min, x_max = max(min(x_coords)-PAD, 0), min(max(x_coords)+PAD, w)
            y_min, y_max = max(min(y_coords)-PAD, 0), min(max(y_coords)+PAD, h)

            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size != 0:
                debug_img = cv2.resize(hand_roi, (64, 64))
                cv2.imwrite('debug_webcam.jpg', debug_img)
                print("📸 Image sauvegardée: debug_webcam.jpg")
                x = preprocess_image_bgr(hand_roi, img_size=64)
                preds = model.predict(x, verbose=0)

# DEBUG: Afficher les top prédictions
                top_5 = np.argsort(preds[0])[-5:][::-1]
                print("\n" + "="*50)
                print("🔍 TOP 5 PRÉDICTIONS:")
                for i, idx in enumerate(top_5):
                    confidence = preds[0][idx] * 100
                    print(f"  {i+1}. {labels[idx]}: {confidence:.1f}%")

                idx = int(np.argmax(preds))
                prob = float(np.max(preds))
                label = labels[idx]
                print(f"✅ CHOISI: {label} ({prob*100:.1f}%)")
                print("="*50)

                label_buffer.append((label, prob))
                if len(label_buffer) > buffer_len:
                    label_buffer.pop(0)

                # majority vote
                votes = {}
                for L,P in label_buffer:
                    votes[L] = votes.get(L, 0) + 1
                display_label = max(votes.items(), key=lambda x: x[1])[0]
                display_prob = max(p for l,p in label_buffer if l==display_label)
                display_text = f"{display_label} {display_prob*100:.1f}%"

            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.imshow('ASL - Mediapipe', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()