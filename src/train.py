import os
import json
import argparse
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_simple_cnn(input_shape=(64,64,3), num_classes=29):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='../dataset', help='dataset root (train/ valid/ test)')
    p.add_argument('--models_dir', default='../models', help='where to save model')
    p.add_argument('--img_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')

    os.makedirs(args.models_dir, exist_ok=True)

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    valid_gen = ImageDataGenerator(rescale=1./255)

    train = train_gen.flow_from_directory(
        train_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    valid = valid_gen.flow_from_directory(
        valid_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    # Solution simple : utiliser le nombre de classes de la validation
    num_classes = valid.num_classes
    print(f"🔧 Utilisation de {num_classes} classes (basé sur validation)")
    
    # Réinitialiser le générateur d'entraînement avec le bon nombre de classes
    train = train_gen.flow_from_directory(
        train_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        classes=valid.class_indices.keys()  # Utiliser les mêmes classes que la validation
    )
    
    model = build_simple_cnn(input_shape=(args.img_size, args.img_size, 3), num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    ckpt_path = os.path.join(args.models_dir, f'asl_model_{timestamp}.h5')

    cb = [
        callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_accuracy', mode='max'),
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]

    history = model.fit(train, validation_data=valid, epochs=args.epochs, callbacks=cb)

    # save final model (best was saved by checkpoint)
    final_path = os.path.join(args.models_dir, 'asl_model_latest.h5')
    model.save(final_path)

    # save history and class indices
    hist_path = os.path.join(args.models_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

    # save class_indices
    class_map = {v: k for k, v in train.class_indices.items()}  # index -> class
    with open(os.path.join(args.models_dir, 'labels.json'), 'w') as f:
        json.dump(class_map, f)

    print('Training finished. Model saved to:', final_path)


if __name__ == '__main__':
    main()