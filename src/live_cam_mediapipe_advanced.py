import cv2
import mediapipe as mp
import numpy as np
from utils import load_trained_model, load_labels, preprocess_image_bgr
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = 'models/asl_model_latest.h5'
PAD = 70

# Dictionnaires de traduction
TRANSLATIONS = {
    'en': {  # English
        'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 
        'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N',
        'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U',
        'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z', 
        'space': ' ', 'del': '⌫', 'nothing': ''
    },
    'fr': {  # Français
        'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 
        'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N',
        'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U',
        'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
        'space': ' ', 'del': '⌫', 'nothing': ''
    },
    'ar': {
    # ASL Letter → Arabic Letter
    'A': 'ا',    # Alif
    'B': 'ب',    # Ba
    'C': 'س',    # Seen
    'D': 'د',    # Dal
    'E': 'ي',    # Ya (pour le son E)
    'F': 'ف',    # Fa
    'G': 'ج',    # Jeem
    'H': 'ه',    # Ha
    'I': 'ي',    # Ya
    'J': 'ج',    # Jeem
    'K': 'ك',    # Kaf
    'L': 'ل',    # Lam
    'M': 'م',    # Meem
    'N': 'ن',    # Noon
    'O': 'و',    # Waw (pour le son O)
    'P': 'ب',    # Ba (pas de P en arabe)
    'Q': 'ق',    # Qaf
    'R': 'ر',    # Ra
    'S': 'ص',    # Sad
    'T': 'ت',    # Ta
    'U': 'و',    # Waw (pour le son U)
    'V': 'ف',    # Fa (pas de V en arabe)
    'W': 'و',    # Waw
    'X': 'كس',   # Kaf-Seen
    'Y': 'ي',    # Ya
    'Z': 'ز',    # Zay
    'space': ' ', 
    'del': '⌫', 
    'nothing': ''
}
}

def draw_arabic_text(image, text, position, font_size=30, color=(255, 255, 0)):
    """Dessiner du texte arabe sur une image OpenCV"""
    try:
        # Convertir BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(image_pil)
        
        # Essayer différentes polices
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",      # Windows Arial
            "C:/Windows/Fonts/tahoma.ttf",     # Windows Tahoma  
            "C:/Windows/Fonts/segoeui.ttf",    # Windows Segoe UI
            "C:/Windows/Fonts/microsoftsansserif.ttf",  # Windows MS Sans
        ]
        
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                print(f"✅ Police chargée: {font_path}")
                break
            except Exception as e:
                continue
                
        if font is None:
            # Police par défaut (moins bonne pour l'arabe)
            font = ImageFont.load_default()
            print("⚠️  Police par défaut utilisée")
        
        # Dessiner le texte arabe
        draw.text(position, text, font=font, fill=color)
        
        # Convertir back to BGR
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        print(f"❌ Erreur affichage arabe: {e}")
        return image

def main():
    model = load_trained_model(MODEL_PATH)
    labels = load_labels(models_dir='models')

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    # Variables pour la construction de mots
    current_word = ""
    label_buffer = []
    buffer_len = 8
    confidence_threshold = 0.7
    last_prediction = None
    language = 'en'  # Par défaut: anglais
    frame_count = 0
    cooldown_frames = 0

    print("🎯 SYSTÈME ASL AVANCÉ ACTIVÉ")
    print("📝 Touches:")
    print("  - ESPACE: Ajouter la lettre détectée")
    print("  - C: Effacer le mot")
    print("  - D: Supprimer dernière lettre") 
    print("  - F: Français / A: Arabe / E: Anglais")
    print("  - Q: Quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Gestion du cooldown pour éviter les détections multiples
        if cooldown_frames > 0:
            cooldown_frames -= 1

        current_prediction = None
        current_confidence = 0

        if results.multi_hand_landmarks and cooldown_frames == 0:
            lm = results.multi_hand_landmarks[0]
            x_coords = [int(p.x * w) for p in lm.landmark]
            y_coords = [int(p.y * h) for p in lm.landmark]
            
            bbox_size = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
            x_center = (min(x_coords) + max(x_coords)) // 2
            y_center = (min(y_coords) + max(y_coords)) // 2
            
            x_min = max(x_center - bbox_size//2 - PAD, 0)
            x_max = min(x_center + bbox_size//2 + PAD, w)
            y_min = max(y_center - bbox_size//2 - PAD, 0)
            y_max = min(y_center + bbox_size//2 + PAD, h)

            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size != 0:
                x = preprocess_image_bgr(hand_roi, img_size=64)
                preds = model.predict(x, verbose=0)

                idx = int(np.argmax(preds))
                prob = float(np.max(preds))
                label = labels[idx]

                # Afficher les prédictions dans le terminal seulement
                if frame_count % 15 == 0:  # Toutes les 15 frames
                    top_3 = np.argsort(preds[0])[-3:][::-1]
                    print(f"\n🔍 Frame {frame_count} - Top 3:")
                    for i, top_idx in enumerate(top_3):
                        conf = preds[0][top_idx] * 100
                        print(f"   {i+1}. {labels[top_idx]}: {conf:.1f}%")

                if prob > confidence_threshold:
                    label_buffer.append((label, prob))
                    if len(label_buffer) > buffer_len:
                        label_buffer.pop(0)

                    # Majority vote
                    if label_buffer:
                        votes = {}
                        for L, P in label_buffer:
                            votes[L] = votes.get(L, 0) + (P * 10)
                        
                        current_prediction = max(votes.items(), key=lambda x: x[1])[0]
                        current_confidence = max(p for l, p in label_buffer if l == current_prediction)

        # Interface utilisateur avancée
        display_text = "Show hand 👋"
        display_color = (0, 0, 255)
        
        if current_prediction and current_confidence > confidence_threshold:
            display_text = f"{current_prediction} ({current_confidence*100:.1f}%)"
            display_color = (0, 255, 0) if current_confidence > 0.8 else (0, 255, 255)
            last_prediction = current_prediction

        # Affichage du mot en cours
        word_display = current_word if current_word else "[Empty]"
        
        # Traduction selon la langue
        translated_word = ""
        for char in current_word:
            if char in TRANSLATIONS[language]:
                translated_word += TRANSLATIONS[language][char]
            else:
                translated_word += char

        # Dessiner l'interface
        cv2.rectangle(frame, (0, 0), (w, 130), (50, 50, 50), -1)
        if language == 'ar':
            frame = draw_arabic_text(frame, f"ترجمة: {translated_word}", (10, 60), 
                                     font_size=20, color=(255, 255, 0))
        else:
            cv2.putText(frame, f"TRADUCTION ({language.upper()}): {translated_word}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Traduction
        cv2.putText(frame, f"TRADUCTION ({language.upper()}): {translated_word}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Lettre détectée
        cv2.putText(frame, f"DETECTED: {display_text}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
        
        # Instructions
        cv2.putText(frame, "SPACE:Add  C:Clear  D:Delete  F/A/E:Language  Q:Quit", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow('ASL - Word Builder', frame)

        # Gestion des touches
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # ESPACE - Ajouter lettre
            if last_prediction and last_prediction != 'nothing':
                if last_prediction == 'space':
                    current_word += ' '
                elif last_prediction == 'del':
                    current_word = current_word[:-1] if current_word else ""
                else:
                    current_word += last_prediction
                print(f"✓ Lettre ajoutée: '{last_prediction}' → Mot: '{current_word}'")
                cooldown_frames = 20  # Cooldown après ajout
                
        elif key == ord('c'):  # C - Clear
            current_word = ""
            print("🗑️ Mot effacé")
            
        elif key == ord('d'):  # D - Delete dernière lettre
            if current_word:
                removed = current_word[-1]
                current_word = current_word[:-1]
                print(f"⌫ Lettre supprimée: '{removed}' → Mot: '{current_word}'")
                
        elif key == ord('f'):  # F - Français
            language = 'fr'
            print("🇫🇷 Langue: Français")
            
        elif key == ord('a'):  # A - Arabe
            language = 'ar' 
            print("🇸🇦 اللغة: العربية")
            
        elif key == ord('e'):  # E - English
            language = 'en'
            print("🇺🇸 Language: English")
            
        elif key == ord('q'):  # Q - Quit
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🎉 Mot final: '{current_word}'")
    print(f"🌍 Traduction ({language}): '{translated_word}'")

if __name__ == '__main__':
    main()