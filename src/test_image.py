import argparse
import cv2
from utils import load_trained_model, load_labels, preprocess_image_bgr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/asl_model_latest.h5')
    p.add_argument('--img', required=True)
    p.add_argument('--labels', default='models/labels.json')
    p.add_argument('--img_size', type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    model = load_trained_model(args.model)
    labels = load_labels(models_dir='models')

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)

    from utils import preprocess_image_bgr
    x = preprocess_image_bgr(img, img_size=args.img_size)
    pred = model.predict(x)
    idx = int(pred.argmax())
    prob = float(pred.max())

    print(f'Prediction: {labels[idx]} ({prob*100:.2f}%)')

if __name__ == '__main__':
    main()