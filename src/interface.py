import cv2
import mediapipe as mp
import numpy as np
from utils import load_trained_model, load_labels, preprocess_image_bgr
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = 'models/asl_model_latest.h5'
PAD = 70

SEMANTIC_TRANSLATIONS = {
    'en': {
        # Pronoms
        'I': 'I', 'YOU': 'You', 'HE': 'He', 'SHE': 'She', 'WE': 'We', 'THEY': 'They', 
        'ME': 'Me', 'MY': 'My', 'YOUR': 'Your', 'OUR': 'Our',
        
        # Salutations / expressions courantes
        'HELLO': 'Hello', 'HI': 'Hi', 'GOODBYE': 'Goodbye', 'BYE': 'Bye',
        'THANKS': 'Thanks', 'THANKYOU': 'Thank you', 'PLEASE': 'Please', 
        'EXCUSEME': 'Excuse me', 'SORRY': 'Sorry', 'STOP': 'Stop', 'GO': 'Go',

        # Questions
        'WHAT': 'What', 'WHERE': 'Where', 'WHEN': 'When', 'WHY': 'Why',
        'HOW': 'How', 'WHO': 'Who',

        # Émotions
        'HAPPY': 'Happy', 'SAD': 'Sad', 'ANGRY': 'Angry', 'LOVE': 'Love', 'LIKE': 'Like',

        # Famille / amis
        'FAMILY': 'Family', 'FRIEND': 'Friend', 'HELP': 'Help',

        # Objets / lieux
        'FOOD': 'Food', 'WATER': 'Water', 'SLEEP': 'Sleep', 'WORK': 'Work',
        'SCHOOL': 'School', 'HOME': 'Home', 'MONEY': 'Money', 'DOG': 'Dog', 'CAT': 'Cat',

        # Verbe “to be” complet
        'AM': 'am', 'IS': 'is', 'ARE': 'are', 'WAS': 'was', 'WERE': 'were', 'BE': 'be', 'BEING': 'being', 'BEEN': 'been',

        # Verbes courants
        'HAVE': 'have', 'HAS': 'has', 'DO': 'do', 'DOES': 'does', 'GOING': 'going',
        'WORKING': 'working', 'STUDYING': 'studying', 'READING': 'reading',
        'WRITING': 'writing', 'PLAYING': 'playing', 'WATCHING': 'watching',

        # Mots pour phrases sur projet
        'THIS': 'this', 'MY': 'my', 'PROJECT': 'project', 'GOOD': 'good', 'BAD': 'bad',

        # Mots supplémentaires utiles
        'TODAY': 'today', 'TOMORROW': 'tomorrow', 'YESTERDAY': 'yesterday', 'NOW': 'now',
        'FAST': 'fast', 'SLOW': 'slow', 'BIG': 'big', 'SMALL': 'small', 'HOT': 'hot', 'COLD': 'cold'
    },

    'fr': {
        # Pronoms
        'I': 'Je', 'YOU': 'Tu', 'HE': 'Il', 'SHE': 'Elle', 'WE': 'Nous', 'THEY': 'Ils',
        'ME': 'Moi', 'MY': 'Mon', 'YOUR': 'Ton', 'OUR': 'Notre',

        # Salutations / expressions
        'HELLO': 'Bonjour', 'HI': 'Salut', 'GOODBYE': 'Au revoir', 'BYE': 'Salut',
        'THANKS': 'Merci', 'THANKYOU': 'Merci', 'PLEASE': 'S il vous plaît',
        'EXCUSEME': 'Excusez-moi', 'SORRY': 'Désolé', 'STOP': 'Arrête', 'GO': 'Aller',

        # Questions
        'WHAT': 'Quoi', 'WHERE': 'Où', 'WHEN': 'Quand', 'WHY': 'Pourquoi',
        'HOW': 'Comment', 'WHO': 'Qui',

        # Émotions
        'HAPPY': 'Heureux', 'SAD': 'Triste', 'ANGRY': 'En colère', 'LOVE': 'Aimer', 'LIKE': 'Aimer',

        # Famille / amis
        'FAMILY': 'Famille', 'FRIEND': 'Ami', 'HELP': 'Aider',

        # Objets / lieux
        'FOOD': 'Nourriture', 'WATER': 'Eau', 'SLEEP': 'Dormir', 'WORK': 'Travail',
        'SCHOOL': 'École', 'HOME': 'Maison', 'MONEY': 'Argent', 'DOG': 'Chien', 'CAT': 'Chat',

        # Verbe “to be” complet
        'AM': 'suis', 'IS': 'est', 'ARE': 'sont', 'WAS': 'était', 'WERE': 'étaient', 'BE': 'être', 'BEING': 'en train d être', 'BEEN': 'été',

        # Verbes courants
        'HAVE': 'avoir', 'HAS': 'a', 'DO': 'faire', 'DOES': 'fait', 'GOING': 'en train d aller',
        'WORKING': 'en train de travailler', 'STUDYING': 'en train d étudier',
        'READING': 'en train de lire', 'WRITING': 'en train d écrire', 'PLAYING': 'en train de jouer', 'WATCHING': 'en train de regarder',

        # Mots pour phrases sur projet
        'THIS': 'ce', 'MY': 'mon', 'PROJECT': 'projet', 'GOOD': 'bon', 'BAD': 'mauvais',

        # Mots supplémentaires utiles
        'TODAY': 'aujourd hui', 'TOMORROW': 'demain', 'YESTERDAY': 'hier', 'NOW': 'maintenant',
        'FAST': 'rapide', 'SLOW': 'lent', 'BIG': 'grand', 'SMALL': 'petit', 'HOT': 'chaud', 'COLD': 'froid'
    },

    'ar': {
        # Pronoms
        'I': 'أنا', 'YOU': 'أنت', 'HE': 'هو', 'SHE': 'هي', 'WE': 'نحن', 'THEY': 'هم',
        'ME': 'أنا', 'MY': 'لي', 'YOUR': 'لك', 'OUR': 'لنا',

        # Salutations / expressions
        'HELLO': 'مرحبا', 'HI': 'أهلا', 'GOODBYE': 'مع السلامة', 'BYE': 'وداعا',
        'THANKS': 'شكرا', 'THANKYOU': 'شكرا لك', 'PLEASE': 'من فضلك',
        'EXCUSEME': 'عفوا', 'SORRY': 'آسف', 'STOP': 'توقف', 'GO': 'اذهب',

        # Questions
        'WHAT': 'ماذا', 'WHERE': 'أين', 'WHEN': 'متى', 'WHY': 'لماذا',
        'HOW': 'كيف', 'WHO': 'من',

        # Émotions
        'HAPPY': 'سعيد', 'SAD': 'حزين', 'ANGRY': 'غاضب', 'LOVE': 'يحب', 'LIKE': 'يحب',

        # Famille / amis
        'FAMILY': 'عائلة', 'FRIEND': 'صديق', 'HELP': 'يساعد',

        # Objets / lieux
        'FOOD': 'طعام', 'WATER': 'ماء', 'SLEEP': 'نوم', 'WORK': 'عمل',
        'SCHOOL': 'مدرسة', 'HOME': 'بيت', 'MONEY': 'مال', 'DOG': 'كلب', 'CAT': 'قط',

        # Verbe “to be” complet
        'AM': 'يكون', 'IS': 'يكون', 'ARE': 'يكونون', 'WAS': 'كان', 'WERE': 'كانوا', 'BE': 'يكون', 'BEING': 'كون', 'BEEN': 'كان',

        # Verbes courants
        'HAVE': 'يمتلك', 'HAS': 'يمتلك', 'DO': 'يفعل', 'DOES': 'يفعل', 'GOING': 'ذاهب',
        'WORKING': 'يعمل', 'STUDYING': 'يدرس', 'READING': 'يقرأ', 'WRITING': 'يكتب',
        'PLAYING': 'يلعب', 'WATCHING': 'يشاهد',

        # Mots pour phrases sur projet
        'THIS': 'هذا', 'MY': 'مشروعي', 'PROJECT': 'مشروع', 'GOOD': 'جيد', 'BAD': 'سيء',

        # Mots supplémentaires utiles
        'TODAY': 'اليوم', 'TOMORROW': 'غدا', 'YESTERDAY': 'أمس', 'NOW': 'الآن',
        'FAST': 'سريع', 'SLOW': 'بطيء', 'BIG': 'كبير', 'SMALL': 'صغير', 'HOT': 'حار', 'COLD': 'بارد'
    }

}

def detect_semantic_words(text):
    """Détection optimisée des mots sémantiques"""
    text_upper = text.upper().replace(' ', '')
    detected_words = []
    remaining_text = text_upper
    
    sorted_words = sorted(SEMANTIC_TRANSLATIONS['en'].keys(), key=len, reverse=True)
    
    for word in sorted_words:
        if word in remaining_text:
            detected_words.append(word)
            remaining_text = remaining_text.replace(word, '', 1)
    
    return detected_words, remaining_text

def smart_translation(current_word, language, translation_mode):
    """Traduction intelligente optimisée"""
    if not translation_mode or not current_word.strip():
        return current_word
    
    detected_words, remaining = detect_semantic_words(current_word)
    
    if detected_words:
        translated_parts = []
        for word in detected_words:
            if word in SEMANTIC_TRANSLATIONS.get(language, {}):
                translated_parts.append(SEMANTIC_TRANSLATIONS[language][word])
            else:
                translated_parts.append(word)
        
        if remaining:
            translated_parts.append(remaining)
        
        return ' '.join(translated_parts)
    else:
        return current_word

def draw_arabic_text(image, text, position, font_size=30, color=(255, 255, 255)):
    """Dessiner du texte arabe sur une image OpenCV"""
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(image_pil)
        
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/tahoma.ttf",  
            "C:/Windows/Fonts/segoeui.ttf",
        ]
        
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
                
        if font is None:
            font = ImageFont.load_default()
        
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        return image

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=15):
    """Dessine un rectangle avec coins arrondis"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_button(frame, text, x, y, w, h, color, text_color, border_color=None):
    """Dessine un bouton moderne avec bordure arrondie"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    if border_color:
        draw_rounded_rectangle(frame, (x, y), (x + w, y + h), border_color, 2, 8)
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

def draw_split_ui(camera_frame, current_word, semantic_translation, language, current_prediction, 
                  current_confidence, translation_mode, hand_landmarks=None, auto_add_mode=False, 
                  stable_frames=0, required_stable_frames=15):
    """Interface split-screen: Camera à gauche, Controls à droite"""
    
    CAMERA_WIDTH = 640
    PANEL_WIDTH = 400
    TOTAL_WIDTH = CAMERA_WIDTH + PANEL_WIDTH
    HEIGHT = 600
    
    canvas = np.zeros((HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (15, 15, 20)  # Fond noir profond
    
    # Couleurs
    pink_primary = (140, 20, 252)
    pink_light = (200, 150, 255)
    purple_accent = (100, 50, 200)
    white = (255, 255, 255)
    gray = (120, 120, 130)
    gray_dark = (50, 50, 60)
    
    # on redimensionner le frame camera
    cam_resized = cv2.resize(camera_frame, (CAMERA_WIDTH, HEIGHT))
    
    # Dessiner les hand landmarks sur la zone camera
    if hand_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        hand_spec = mp_drawing.DrawingSpec(color=(140, 20, 252), thickness=3, circle_radius=3)
        connection_spec = mp_drawing.DrawingSpec(color=(200, 150, 255), thickness=2)
        
        mp_drawing.draw_landmarks(
            cam_resized, hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            hand_spec, connection_spec
        )
    
    # bordure rose autour de la camera
    cv2.rectangle(cam_resized, (0, 0), (CAMERA_WIDTH-1, HEIGHT-1), pink_primary, 3)
    
    # placer la camera sur le canvas
    canvas[0:HEIGHT, 0:CAMERA_WIDTH] = cam_resized
    
    panel_x = CAMERA_WIDTH
    
    # Background noir pour le panel
    cv2.rectangle(canvas, (panel_x, 0), (TOTAL_WIDTH, HEIGHT), (15, 15, 20), -1)
    
    # Ligne rose
    cv2.rectangle(canvas, (panel_x, 0), (panel_x + 3, HEIGHT), pink_primary, -1)
    cv2.putText(canvas, "ASL", (panel_x + 20, 40), 
               cv2.FONT_HERSHEY_DUPLEX, 1.1, white, 2, cv2.LINE_AA)
    cv2.putText(canvas, "SILENT DIALOGUE", (panel_x + 20, 70), 
               cv2.FONT_HERSHEY_DUPLEX, 0.8, pink_light, 2, cv2.LINE_AA)
    
    # Ligne décorative
    cv2.line(canvas, (panel_x + 20, 85), (panel_x + 360, 85), pink_primary, 2)
    
    status_y = 110
    
    if current_prediction and current_confidence > 0:
        if current_confidence > 0.85:
            badge_color = pink_primary
            status_text = "EXCELLENT"
        elif current_confidence > 0.70:
            badge_color = purple_accent
            status_text = "GOOD"
        else:
            badge_color = gray
            status_text = "LOW"
        
        # Card detection
        cv2.rectangle(canvas, (panel_x + 20, status_y), 
                     (panel_x + 360, status_y + 70), gray_dark, -1)
        draw_rounded_rectangle(canvas, (panel_x + 20, status_y), 
                              (panel_x + 360, status_y + 70), badge_color, 2, 10)
        
        cv2.putText(canvas, "DETECTED", (panel_x + 30, status_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, gray, 1, cv2.LINE_AA)
        cv2.putText(canvas, current_prediction.upper(), (panel_x + 30, status_y + 48), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, white, 2, cv2.LINE_AA)
        conf_text = f"{current_confidence*100:.0f}% - {status_text}"
        cv2.putText(canvas, conf_text, (panel_x + 30, status_y + 64), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, pink_light, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (panel_x + 20, status_y), 
                     (panel_x + 360, status_y + 70), gray_dark, -1)
        draw_rounded_rectangle(canvas, (panel_x + 20, status_y), 
                              (panel_x + 360, status_y + 70), gray, 1, 10)
        
        cv2.putText(canvas, "WAITING FOR SIGN...", (panel_x + 30, status_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, gray, 1, cv2.LINE_AA)
    
    letters_y = status_y + 90
    cv2.rectangle(canvas, (panel_x + 20, letters_y), 
                 (panel_x + 360, letters_y + 80), gray_dark, -1)
    draw_rounded_rectangle(canvas, (panel_x + 20, letters_y), 
                          (panel_x + 360, letters_y + 80), pink_light, 2, 10)
    
    cv2.putText(canvas, "LETTERS", (panel_x + 30, letters_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, gray, 1, cv2.LINE_AA)
    
    letters_text = current_word if current_word else "..."
    if len(letters_text) > 20:
        letters_text = letters_text[:17] + "..."
    
    cv2.putText(canvas, letters_text, (panel_x + 30, letters_y + 55), 
               cv2.FONT_HERSHEY_DUPLEX, 0.65, white, 2, cv2.LINE_AA)
    
    # Compteur
    char_count = f"{len(current_word)} chars"
    cv2.putText(canvas, char_count, (panel_x + 30, letters_y + 72), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, gray, 1, cv2.LINE_AA)
    
    # =============== TRANSLATION CARD ===============
    trans_y = letters_y + 95
    
    border_color = pink_primary if translation_mode else gray
    cv2.rectangle(canvas, (panel_x + 20, trans_y), 
                 (panel_x + 360, trans_y + 80), gray_dark, -1)
    draw_rounded_rectangle(canvas, (panel_x + 20, trans_y), 
                          (panel_x + 360, trans_y + 80), border_color, 2, 10)
    
    mode_label = "TRANSLATION (ON)" if translation_mode else "TRANSLATION (OFF)"
    cv2.putText(canvas, mode_label, (panel_x + 30, trans_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, gray, 1, cv2.LINE_AA)
    
    if translation_mode and current_word:
        translation_text = semantic_translation if semantic_translation else current_word
        if len(translation_text) > 20:
            translation_text = translation_text[:17] + "..."
        
        if language == 'ar':
            canvas = draw_arabic_text(canvas, translation_text, 
                                    (panel_x + 30, trans_y + 42), font_size=16, color=pink_light)
        else:
            cv2.putText(canvas, translation_text, (panel_x + 30, trans_y + 55), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.65, pink_light, 2, cv2.LINE_AA)
    else:
        placeholder = "Disabled" if not translation_mode else "..."
        cv2.putText(canvas, placeholder, (panel_x + 30, trans_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gray, 1, cv2.LINE_AA)
    
    # =============== CONTROLS ===============
    ctrl_y = trans_y + 95
    btn_w = 340
    btn_h = 32
    
    # Auto-Add
    auto_bg = pink_primary if auto_add_mode else gray_dark
    auto_border = pink_primary if auto_add_mode else gray
    auto_text = "AUTO-ADD: ON" if auto_add_mode else "AUTO-ADD: OFF"
    draw_button(canvas, auto_text, panel_x + 20, ctrl_y, btn_w, btn_h, auto_bg, white, auto_border)
    
    # Progress bar
    if auto_add_mode and stable_frames > 0 and current_prediction:
        progress = min(stable_frames / required_stable_frames, 1.0)
        bar_w = int(btn_w * progress)
        cv2.rectangle(canvas, (panel_x + 20, ctrl_y + btn_h - 3), 
                     (panel_x + 20 + bar_w, ctrl_y + btn_h), pink_primary, -1)
    
    # Translation Mode
    trans_bg = pink_primary if translation_mode else gray_dark
    draw_button(canvas, "TRANSLATION MODE", panel_x + 20, ctrl_y + 40, 
               btn_w, btn_h, trans_bg, white, pink_light if translation_mode else gray)
    
    # Languages (3 boutons côte à côte)
    lang_names = {'en': 'EN', 'fr': 'FR', 'ar': 'AR'}
    lang_y = ctrl_y + 80
    lang_btn_w = 105
    
    for i, (lang_code, lang_name) in enumerate(lang_names.items()):
        x_pos = panel_x + 20 + i * (lang_btn_w + 12)
        is_active = language == lang_code
        btn_bg = pink_primary if is_active else gray_dark
        btn_border = pink_primary if is_active else gray
        draw_button(canvas, lang_name, x_pos, lang_y, lang_btn_w, btn_h, 
                   btn_bg, white if is_active else gray, btn_border)
    
    # =============== FOOTER CONTROLS ===============
    footer_y = HEIGHT - 50
    cv2.rectangle(canvas, (panel_x, footer_y), (TOTAL_WIDTH, HEIGHT), (10, 10, 15), -1)
    cv2.line(canvas, (panel_x, footer_y), (TOTAL_WIDTH, footer_y), pink_primary, 2)
    
    # Instructions compactes
    controls_text = "SPC:Add  ⌫:Del  C:Clear  M:Auto  T:Trans  E/F/A:Lang  S:Save  Q:Quit"
    cv2.putText(canvas, controls_text, (panel_x + 15, footer_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, pink_light, 1, cv2.LINE_AA)
    
    return canvas

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

    current_word = ""
    semantic_translation = ""
    label_buffer = []
    buffer_len = 8
    confidence_threshold = 0.7
    last_prediction = None
    language = 'en'
    frame_count = 0
    cooldown_frames = 0
    translation_mode = True
    saved_texts = []
    auto_add_mode = False
    last_added_prediction = None
    stable_frames = 0
    required_stable_frames = 15

    print("🎨 ASL Interpreter - Split Screen UI")
    print("✨ Camera Left | Controls Right")
    print("⌫ BACKSPACE to delete | M for AUTO-ADD")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if cooldown_frames > 0:
            cooldown_frames -= 1

        current_prediction = None
        current_confidence = 0
        hand_landmarks_to_draw = None

        if results.multi_hand_landmarks and cooldown_frames == 0:
            lm = results.multi_hand_landmarks[0]
            hand_landmarks_to_draw = lm
            
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

                if prob > confidence_threshold:
                    label_buffer.append((label, prob))
                    if len(label_buffer) > buffer_len:
                        label_buffer.pop(0)
                    if label_buffer:
                        votes = {}
                        for L, P in label_buffer:
                            votes[L] = votes.get(L, 0) + (P * 10)
                        current_prediction = max(votes.items(), key=lambda x: x[1])[0]
                        current_confidence = max(p for l, p in label_buffer if l == current_prediction)
                        
                        # Auto-add mode
                        if auto_add_mode and current_prediction != 'nothing':
                            if current_prediction == last_added_prediction:
                                stable_frames += 1
                            else:
                                stable_frames = 0
                                last_added_prediction = current_prediction
                            
                            if stable_frames >= required_stable_frames and current_confidence > 0.8:
                                if current_prediction == 'space':
                                    current_word += ' '
                                elif current_prediction == 'del':
                                    current_word = current_word[:-1] if current_word else ""
                                else:
                                    current_word += current_prediction
                                stable_frames = 0
                                last_added_prediction = None
                                cooldown_frames = 10
                        else:
                            stable_frames = 0
                            last_added_prediction = None

        semantic_translation = smart_translation(current_word, language, translation_mode)

        display = draw_split_ui(frame, current_word, semantic_translation, language, 
                               current_prediction, current_confidence, translation_mode, 
                               hand_landmarks_to_draw, auto_add_mode, stable_frames, required_stable_frames)
        
        cv2.imshow('ASL Interpreter Pro - Split View', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if current_prediction and current_prediction != 'nothing':
                if current_prediction == 'space':
                    current_word += ' '
                elif current_prediction == 'del':
                    current_word = current_word[:-1] if current_word else ""
                else:
                    current_word += current_prediction
                last_prediction = current_prediction
                cooldown_frames = 20
        elif key == 8 or key == 127:  # BACKSPACE
            if current_word:
                current_word = current_word[:-1]
                print(f"⌫ Deleted: '{current_word}'")
        elif key == ord('c'):
            current_word = ""
            semantic_translation = ""
            print("🗑️ Cleared")
        elif key == ord('d') and current_word:
            current_word = current_word[:-1]
        elif key == ord('f'):
            language = 'fr'
            print(f"🇫🇷 Français")
        elif key == ord('a'):
            language = 'ar'
            print(f"🇸🇦 العربية")
        elif key == ord('e'):
            language = 'en'
            print(f"🇬🇧 English")
        elif key == ord('t'):
            translation_mode = not translation_mode
            print(f"🔤 Translation: {'ON' if translation_mode else 'OFF'}")
        elif key == ord('m'):
            auto_add_mode = not auto_add_mode
            print(f"⚡ Auto-Add: {'ON' if auto_add_mode else 'OFF'}")
            stable_frames = 0
            last_added_prediction = None
        elif key == ord('s'):
            if current_word:
                saved_texts.append({
                    'text': current_word,
                    'translation': semantic_translation,
                    'language': language
                })
                print(f"💾 Saved: {current_word} → {semantic_translation}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if saved_texts:
        print("\n📝 Saved Texts:")
        for i, item in enumerate(saved_texts, 1):
            print(f"  {i}. {item['text']} → {item['translation']} ({item['language']})")
    
    print("👋 Closed")

if __name__ == '__main__':
    main()