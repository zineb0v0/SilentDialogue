import cv2
import mediapipe as mp
import numpy as np
from utils import load_trained_model, load_labels, preprocess_image_bgr
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = 'models/asl_model_latest.h5'
PAD = 70

# DICTIONNAIRE OPTIMISÉ (150 MOTS FRÉQUENTS)
SEMANTIC_TRANSLATIONS = {
    'en': {
        # ========== PRONOMS & BASIQUES ==========
        'I': 'I', 'YOU': 'You', 'HE': 'He', 'SHE': 'She', 'WE': 'We', 
        'THEY': 'They', 'ME': 'Me', 'MY': 'My', 'YOUR': 'Your', 'OUR': 'Our',
        'THEIR': 'Their', 'HIS': 'His', 'HER': 'Her', 'IT': 'It', 'ITS': 'Its',
        
        # ========== VERBES COURANTS ==========
        'BE': 'Be', 'IS': 'Is', 'ARE': 'Are', 'AM': 'Am', 'HAVE': 'Have',
        'HAS': 'Has', 'DO': 'Do', 'DONT': "Don't", 'CAN': 'Can', 'CANNOT': 'Cannot',
        'WILL': 'Will', 'WOULD': 'Would', 'SHOULD': 'Should', 'COULD': 'Could',
        'MAY': 'May', 'MIGHT': 'Might', 'MUST': 'Must', 'WANT': 'Want',
        'NEED': 'Need', 'LIKE': 'Like', 'LOVE': 'Love', 'HATE': 'Hate',
        
        # ========== SALUTATIONS & RÉPONSES ==========
        'HELLO': 'Hello', 'HI': 'Hi', 'GOODBYE': 'Goodbye', 'BYE': 'Bye',
        'WELCOME': 'Welcome', 'THANKS': 'Thanks', 'THANKYOU': 'Thank you',
        'PLEASE': 'Please', 'SORRY': 'Sorry', 'EXCUSEME': 'Excuse me',
        'YES': 'Yes', 'NO': 'No', 'OK': 'OK', 'FINE': 'Fine', 'GOOD': 'Good',
        'BAD': 'Bad', 'WELL': 'Well', 'GREAT': 'Great', 'PERFECT': 'Perfect',
        'MAYBE': 'Maybe', 'SURE': 'Sure', 'NOT': 'Not',
        
        # ========== QUESTIONS ==========
        'WHAT': 'What', 'WHERE': 'Where', 'WHEN': 'When', 'WHY': 'Why',
        'HOW': 'How', 'WHO': 'Who', 'WHICH': 'Which', 'WHOSE': 'Whose',
        
        # ========== FAMILLE & PERSONNES ==========
        'FAMILY': 'Family', 'FATHER': 'Father', 'MOTHER': 'Mother', 
        'PARENTS': 'Parents', 'SON': 'Son', 'DAUGHTER': 'Daughter',
        'BROTHER': 'Brother', 'SISTER': 'Sister', 'BABY': 'Baby',
        'CHILD': 'Child', 'MAN': 'Man', 'WOMAN': 'Woman', 'BOY': 'Boy',
        'GIRL': 'Girl', 'FRIEND': 'Friend', 'NAME': 'Name',
        
        # ========== ÉMOTIONS & SENTIMENTS ==========
        'HAPPY': 'Happy', 'SAD': 'Sad', 'ANGRY': 'Angry', 'EXCITED': 'Excited',
        'SCARED': 'Scared', 'CALM': 'Calm', 'TIRED': 'Tired', 'SICK': 'Sick',
        
        # ========== LIEUX & MAISON ==========
        'HOME': 'Home', 'HOUSE': 'House', 'ROOM': 'Room', 'BATHROOM': 'Bathroom',
        'BEDROOM': 'Bedroom', 'KITCHEN': 'Kitchen', 'SCHOOL': 'School',
        'WORK': 'Work', 'OFFICE': 'Office', 'HOSPITAL': 'Hospital',
        'PARK': 'Park', 'STORE': 'Store', 'CITY': 'City',
        
        # ========== NOURRITURE & BOISSONS ==========
        'FOOD': 'Food', 'EAT': 'Eat', 'DRINK': 'Drink', 'WATER': 'Water',
        'HUNGRY': 'Hungry', 'THIRSTY': 'Thirsty', 'BREAD': 'Bread',
        'MEAT': 'Meat', 'FRUIT': 'Fruit', 'MILK': 'Milk', 'COFFEE': 'Coffee',
        
        # ========== ACTIONS & MOUVEMENTS ==========
        'GO': 'Go', 'COME': 'Come', 'SEE': 'See', 'LOOK': 'Look',
        'HEAR': 'Hear', 'LISTEN': 'Listen', 'SPEAK': 'Speak', 'TALK': 'Talk',
        'SAY': 'Say', 'ASK': 'Ask', 'ANSWER': 'Answer', 'THINK': 'Think',
        'KNOW': 'Know', 'UNDERSTAND': 'Understand', 'REMEMBER': 'Remember',
        'FORGET': 'Forget', 'HELP': 'Help', 'STOP': 'Stop', 'START': 'Start',
        'WAIT': 'Wait', 'SIT': 'Sit', 'STAND': 'Stand', 'WALK': 'Walk',
        'RUN': 'Run', 'SLEEP': 'Sleep', 'WAKE': 'Wake',
        
        # ========== OBJETS & COULEURS ==========
        'BOOK': 'Book', 'PEN': 'Pen', 'PAPER': 'Paper', 'PHONE': 'Phone',
        'COMPUTER': 'Computer', 'CAR': 'Car', 'BUS': 'Bus', 'TRAIN': 'Train',
        'BED': 'Bed', 'TABLE': 'Table', 'CHAIR': 'Chair', 'DOOR': 'Door',
        'WINDOW': 'Window', 'RED': 'Red', 'BLUE': 'Blue', 'GREEN': 'Green',
        'YELLOW': 'Yellow', 'BLACK': 'Black', 'WHITE': 'White',
        
        # ========== TEMPS & NOMBRES ==========
        'TIME': 'Time', 'DAY': 'Day', 'NIGHT': 'Night', 'TODAY': 'Today',
        'TOMORROW': 'Tomorrow', 'NOW': 'Now', 'LATER': 'Later',
        'ONE': 'One', 'TWO': 'Two', 'THREE': 'Three', 'FOUR': 'Four',
        'FIVE': 'Five', 'TEN': 'Ten',
        
        # ========== PHRASES COURANTES ==========
        'ILOVEYOU': 'I love you', 'HOWAREYOU': 'How are you',
        'WHATISYOURNAME': 'What is your name', 'MYNAMEIS': 'My name is',
        'NICETOMEETYOU': 'Nice to meet you', 'WHEREAREYOUFROM': 'Where are you from',
        'CANYOUHELPME': 'Can you help me', 'IDONTUNDERSTAND': "I don't understand",
        'GOODMORNING': 'Good morning', 'GOODNIGHT': 'Good night',
        'SEEYOULATER': 'See you later', 'HAVEANICEDAY': 'Have a nice day'
    },
    
    'fr': {
        # Version française
        'I': 'Je', 'YOU': 'Tu', 'HE': 'Il', 'SHE': 'Elle', 'WE': 'Nous',
        'THEY': 'Ils', 'ME': 'Moi', 'MY': 'Mon', 'YOUR': 'Ton', 'OUR': 'Notre',
        'THEIR': 'Leur', 'HIS': 'Son', 'HER': 'Sa', 'IT': 'Il', 'ITS': 'Son',
        
        'BE': 'Être', 'IS': 'Est', 'ARE': 'Sont', 'AM': 'Suis', 'HAVE': 'Avoir',
        'HAS': 'A', 'DO': 'Faire', 'DONT': 'Ne pas', 'CAN': 'Pouvoir',
        'CANNOT': 'Ne peut pas', 'WILL': 'Vouloir', 'WOULD': 'Voudrait',
        'SHOULD': 'Devrait', 'COULD': 'Pourrait', 'MAY': 'Peut-être',
        'MIGHT': 'Pourrait', 'MUST': 'Doit', 'WANT': 'Vouloir',
        'NEED': 'Avoir besoin', 'LIKE': 'Aimer', 'LOVE': 'Aimer', 'HATE': 'Détester',
        
        'HELLO': 'Bonjour', 'HI': 'Salut', 'GOODBYE': 'Au revoir', 'BYE': 'Salut',
        'WELCOME': 'Bienvenue', 'THANKS': 'Merci', 'THANKYOU': 'Merci',
        'PLEASE': 'S il vous plaît', 'SORRY': 'Désolé', 'EXCUSEME': 'Excusez-moi',
        'YES': 'Oui', 'NO': 'Non', 'OK': 'D accord', 'FINE': 'Bien', 'GOOD': 'Bon',
        'BAD': 'Mauvais', 'WELL': 'Bien', 'GREAT': 'Génial', 'PERFECT': 'Parfait',
        'MAYBE': 'Peut-être', 'SURE': 'Bien sûr', 'NOT': 'Pas',
        
        'WHAT': 'Quoi', 'WHERE': 'Où', 'WHEN': 'Quand', 'WHY': 'Pourquoi',
        'HOW': 'Comment', 'WHO': 'Qui', 'WHICH': 'Quel', 'WHOSE': 'À qui',
        
        'FAMILY': 'Famille', 'FATHER': 'Père', 'MOTHER': 'Mère',
        'PARENTS': 'Parents', 'SON': 'Fils', 'DAUGHTER': 'Fille',
        'BROTHER': 'Frère', 'SISTER': 'Sœur', 'BABY': 'Bébé',
        'CHILD': 'Enfant', 'MAN': 'Homme', 'WOMAN': 'Femme', 'BOY': 'Garçon',
        'GIRL': 'Fille', 'FRIEND': 'Ami', 'NAME': 'Nom',
        
        'HAPPY': 'Heureux', 'SAD': 'Triste', 'ANGRY': 'En colère',
        'EXCITED': 'Excité', 'SCARED': 'Peur', 'CALM': 'Calme',
        'TIRED': 'Fatigué', 'SICK': 'Malade',
        
        'HOME': 'Maison', 'HOUSE': 'Maison', 'ROOM': 'Pièce',
        'BATHROOM': 'Salle de bain', 'BEDROOM': 'Chambre',
        'KITCHEN': 'Cuisine', 'SCHOOL': 'École', 'WORK': 'Travail',
        'OFFICE': 'Bureau', 'HOSPITAL': 'Hôpital', 'PARK': 'Parc',
        'STORE': 'Magasin', 'CITY': 'Ville',
        
        'FOOD': 'Nourriture', 'EAT': 'Manger', 'DRINK': 'Boire',
        'WATER': 'Eau', 'HUNGRY': 'Faim', 'THIRSTY': 'Soif',
        'BREAD': 'Pain', 'MEAT': 'Viande', 'FRUIT': 'Fruit', 'MILK': 'Lait',
        'COFFEE': 'Café',
        
        'GO': 'Aller', 'COME': 'Venir', 'SEE': 'Voir', 'LOOK': 'Regarder',
        'HEAR': 'Entendre', 'LISTEN': 'Écouter', 'SPEAK': 'Parler',
        'TALK': 'Parler', 'SAY': 'Dire', 'ASK': 'Demander',
        'ANSWER': 'Répondre', 'THINK': 'Penser', 'KNOW': 'Savoir',
        'UNDERSTAND': 'Comprendre', 'REMEMBER': 'Se souvenir',
        'FORGET': 'Oublier', 'HELP': 'Aider', 'STOP': 'Arrêter',
        'START': 'Commencer', 'WAIT': 'Attendre', 'SIT': 'S asseoir',
        'STAND': 'Se lever', 'WALK': 'Marcher', 'RUN': 'Courir',
        'SLEEP': 'Dormir', 'WAKE': 'Se réveiller',
        
        'BOOK': 'Livre', 'PEN': 'Stylo', 'PAPER': 'Papier', 'PHONE': 'Téléphone',
        'COMPUTER': 'Ordinateur', 'CAR': 'Voiture', 'BUS': 'Bus',
        'TRAIN': 'Train', 'BED': 'Lit', 'TABLE': 'Table', 'CHAIR': 'Chaise',
        'DOOR': 'Porte', 'WINDOW': 'Fenêtre', 'RED': 'Rouge', 'BLUE': 'Bleu',
        'GREEN': 'Vert', 'YELLOW': 'Jaune', 'BLACK': 'Noir', 'WHITE': 'Blanc',
        
        'TIME': 'Temps', 'DAY': 'Jour', 'NIGHT': 'Nuit', 'TODAY': 'Aujourd hui',
        'TOMORROW': 'Demain', 'NOW': 'Maintenant', 'LATER': 'Plus tard',
        'ONE': 'Un', 'TWO': 'Deux', 'THREE': 'Trois', 'FOUR': 'Quatre',
        'FIVE': 'Cinq', 'TEN': 'Dix',
        
        'ILOVEYOU': 'Je t aime', 'HOWAREYOU': 'Comment allez-vous',
        'WHATISYOURNAME': 'Comment vous appelez-vous', 'MYNAMEIS': 'Je m appelle',
        'NICETOMEETYOU': 'Enchanté de vous rencontrer',
        'WHEREAREYOUFROM': 'D où venez-vous', 'CANYOUHELPME': 'Pouvez-vous m aider',
        'IDONTUNDERSTAND': 'Je ne comprends pas', 'GOODMORNING': 'Bonjour',
        'GOODNIGHT': 'Bonne nuit', 'SEEYOULATER': 'À plus tard',
        'HAVEANICEDAY': 'Bonne journée'
    },
    
    'ar': {
        # Version arabe
        'I': 'أنا', 'YOU': 'أنت', 'HE': 'هو', 'SHE': 'هي', 'WE': 'نحن',
        'THEY': 'هم', 'ME': 'أنا', 'MY': 'لي', 'YOUR': 'لك', 'OUR': 'لنا',
        'THEIR': 'لهم', 'HIS': 'له', 'HER': 'لها', 'IT': 'هو', 'ITS': 'له',
        
        'BE': 'يكون', 'IS': 'هو', 'ARE': 'هم', 'AM': 'أنا', 'HAVE': 'يملك',
        'HAS': 'يملك', 'DO': 'يفعل', 'DONT': 'لا', 'CAN': 'يستطيع',
        'CANNOT': 'لا يستطيع', 'WILL': 'سوف', 'WOULD': 'سوف', 'SHOULD': 'يجب',
        'COULD': 'يمكن', 'MAY': 'قد', 'MIGHT': 'قد', 'MUST': 'يجب',
        'WANT': 'يريد', 'NEED': 'يحتاج', 'LIKE': 'يحب', 'LOVE': 'يحب',
        'HATE': 'يكره',
        
        'HELLO': 'مرحبا', 'HI': 'أهلا', 'GOODBYE': 'مع السلامة', 'BYE': 'وداعا',
        'WELCOME': 'أهلا وسهلا', 'THANKS': 'شكرا', 'THANKYOU': 'شكرا لك',
        'PLEASE': 'من فضلك', 'SORRY': 'آسف', 'EXCUSEME': 'اعذرني',
        'YES': 'نعم', 'NO': 'لا', 'OK': 'موافق', 'FINE': 'بخير', 'GOOD': 'جيد',
        'BAD': 'سيء', 'WELL': 'جيد', 'GREAT': 'عظيم', 'PERFECT': 'ممتاز',
        'MAYBE': 'ربما', 'SURE': 'بالتأكيد', 'NOT': 'ليس',
        
        'WHAT': 'ماذا', 'WHERE': 'أين', 'WHEN': 'متى', 'WHY': 'لماذا',
        'HOW': 'كيف', 'WHO': 'من', 'WHICH': 'أي', 'WHOSE': 'لمن',
        
        'FAMILY': 'عائلة', 'FATHER': 'أب', 'MOTHER': 'أم',
        'PARENTS': 'والدان', 'SON': 'ابن', 'DAUGHTER': 'ابنة',
        'BROTHER': 'أخ', 'SISTER': 'أخت', 'BABY': 'طفل',
        'CHILD': 'طفل', 'MAN': 'رجل', 'WOMAN': 'امرأة', 'BOY': 'ولد',
        'GIRL': 'بنت', 'FRIEND': 'صديق', 'NAME': 'اسم',
        
        'HAPPY': 'سعيد', 'SAD': 'حزين', 'ANGRY': 'غاضب',
        'EXCITED': 'متحمس', 'SCARED': 'خائف', 'CALM': 'هادئ',
        'TIRED': 'متعب', 'SICK': 'مريض',
        
        'HOME': 'بيت', 'HOUSE': 'منزل', 'ROOM': 'غرفة',
        'BATHROOM': 'حمام', 'BEDROOM': 'غرفة نوم', 'KITCHEN': 'مطبخ',
        'SCHOOL': 'مدرسة', 'WORK': 'عمل', 'OFFICE': 'مكتب',
        'HOSPITAL': 'مستشفى', 'PARK': 'حديقة', 'STORE': 'متجر',
        'CITY': 'مدينة',
        
        'FOOD': 'طعام', 'EAT': 'يأكل', 'DRINK': 'يشرب', 'WATER': 'ماء',
        'HUNGRY': 'جوعان', 'THIRSTY': 'عطشان', 'BREAD': 'خبز',
        'MEAT': 'لحم', 'FRUIT': 'فاكهة', 'MILK': 'حليب', 'COFFEE': 'قهوة',
        
        'GO': 'يذهب', 'COME': 'يأتي', 'SEE': 'يرى', 'LOOK': 'ينظر',
        'HEAR': 'يسمع', 'LISTEN': 'يستمع', 'SPEAK': 'يتكلم',
        'TALK': 'يتحدث', 'SAY': 'يقول', 'ASK': 'يسأل', 'ANSWER': 'يجيب',
        'THINK': 'يفكر', 'KNOW': 'يعرف', 'UNDERSTAND': 'يفهم',
        'REMEMBER': 'يتذكر', 'FORGET': 'ينسى', 'HELP': 'يساعد',
        'STOP': 'يتوقف', 'START': 'يبدأ', 'WAIT': 'ينتظر', 'SIT': 'يجلس',
        'STAND': 'يقف', 'WALK': 'يمشي', 'RUN': 'يركض', 'SLEEP': 'ينام',
        'WAKE': 'يستيقظ',
        
        'BOOK': 'كتاب', 'PEN': 'قلم', 'PAPER': 'ورقة', 'PHONE': 'هاتف',
        'COMPUTER': 'كمبيوتر', 'CAR': 'سيارة', 'BUS': 'حافلة',
        'TRAIN': 'قطار', 'BED': 'سرير', 'TABLE': 'طاولة', 'CHAIR': 'كرسي',
        'DOOR': 'باب', 'WINDOW': 'نافذة', 'RED': 'أحمر', 'BLUE': 'أزرق',
        'GREEN': 'أخضر', 'YELLOW': 'أصفر', 'BLACK': 'أسود', 'WHITE': 'أبيض',
        
        'TIME': 'وقت', 'DAY': 'يوم', 'NIGHT': 'ليل', 'TODAY': 'اليوم',
        'TOMORROW': 'غدا', 'NOW': 'الآن', 'LATER': 'لاحقا',
        'ONE': 'واحد', 'TWO': 'اثنان', 'THREE': 'ثلاثة', 'FOUR': 'أربعة',
        'FIVE': 'خمسة', 'TEN': 'عشرة',
        
        'ILOVEYOU': 'أحبك', 'HOWAREYOU': 'كيف حالك',
        'WHATISYOURNAME': 'ما اسمك', 'MYNAMEIS': 'اسمي هو',
        'NICETOMEETYOU': 'تشرفت بلقائك', 'WHEREAREYOUFROM': 'من أين أنت',
        'CANYOUHELPME': 'هل يمكنك مساعدتي', 'IDONTUNDERSTAND': 'أنا لا أفهم',
        'GOODMORNING': 'صباح الخير', 'GOODNIGHT': 'تصبح على خير',
        'SEEYOULATER': 'أراك لاحقا', 'HAVEANICEDAY': 'أتمنى لك يوما سعيدا'
    }
}

def detect_semantic_words(text):
    """Détection optimisée des mots sémantiques"""
    text_upper = text.upper().replace(' ', '')
    detected_words = []
    remaining_text = text_upper
    
    # Chercher les mots du plus long au plus court (optimisation)
    for word_length in range(20, 1, -1):  # Mots de 20 lettres à 2 lettres
        for word in SEMANTIC_TRANSLATIONS['en']:
            if len(word) == word_length and word in remaining_text:
                detected_words.append(word)
                remaining_text = remaining_text.replace(word, '', 1)
    
    return detected_words, remaining_text

def smart_translation(current_word, language, translation_mode):
    """Traduction intelligente optimisée"""
    if not translation_mode or not current_word.strip():
        return current_word
    
    detected_words, remaining = detect_semantic_words(current_word)
    
    if detected_words:
        # Traduire les mots détectés
        translated_parts = []
        for word in detected_words:
            if word in SEMANTIC_TRANSLATIONS[language]:
                translated_parts.append(SEMANTIC_TRANSLATIONS[language][word])
        
        # Ajouter les lettres restantes
        if remaining:
            translated_parts.append(remaining)
        
        return ' '.join(translated_parts)
    else:
        return current_word

def draw_arabic_text(image, text, position, font_size=30, color=(255, 255, 0)):
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

    # Variables
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

    print("🎯 SYSTÈME ASL - DICTIONNAIRE OPTIMISÉ (150 MOTS)")
    print(f"📚 Mots disponibles: {len(SEMANTIC_TRANSLATIONS['en'])}")
    print("⌨️  Touches: ESPACE=Ajouter, C=Effacer, D=Supprimer, F/A/E=Langue, T=Mode, Q=Quitter")

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

                if frame_count % 15 == 0:
                    top_3 = np.argsort(preds[0])[-3:][::-1]
                    print(f"\n🔍 Frame {frame_count}: {labels[top_3[0]]}({preds[0][top_3[0]]*100:.1f}%)")

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

        # TRADUCTION SÉMANTIQUE OPTIMISÉE
        semantic_translation = smart_translation(current_word, language, translation_mode)

        # Interface utilisateur
        display_text = "Show hand 👋"
        display_color = (0, 0, 255)
        
        if current_prediction and current_confidence > confidence_threshold:
            display_text = f"{current_prediction} ({current_confidence*100:.1f}%)"
            display_color = (0, 255, 0) if current_confidence > 0.8 else (0, 255, 255)
            last_prediction = current_prediction

        # Dessiner l'interface
        cv2.rectangle(frame, (0, 0), (w, 150), (50, 50, 50), -1)
        
        # Mot en cours
        cv2.putText(frame, f"LETTERS: {current_word}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Traduction
        mode_text = "🔤 TRADUCTION" if translation_mode else "🔠 LETTRES"
        if language == 'ar':
            frame = draw_arabic_text(frame, f"{mode_text}: {semantic_translation}", 
                                   (10, 50), font_size=18, color=(255, 255, 0))
        else:
            cv2.putText(frame, f"{mode_text}: {semantic_translation}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Détection actuelle
        cv2.putText(frame, f"DETECTED: {display_text}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
        
        # Info mots détectés
        detected_words, _ = detect_semantic_words(current_word)
        if detected_words and translation_mode:
            cv2.putText(frame, f"WORDS: {', '.join(detected_words[:3])}", (10, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "SPACE:Add  C:Clear  D:Del  F/A/E:Lang  T:Mode  Q:Quit", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv2.imshow('ASL - Optimized Dictionary (150 words)', frame)

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
                
                # Détection et affichage des mots
                detected_words, _ = detect_semantic_words(current_word)
                if detected_words and translation_mode:
                    print(f"🎯 Mots détectés: {detected_words}")
                    for word in detected_words:
                        print(f"   → {word}: {SEMANTIC_TRANSLATIONS[language][word]}")
                
                cooldown_frames = 20
                
        elif key == ord('c'):  # C - Effacer
            current_word = ""
            print("🗑️ Mot effacé")
            
        elif key == ord('d') and current_word:  # D - Supprimer
            removed = current_word[-1]
            current_word = current_word[:-1]
            print(f"⌫ Supprimé: '{removed}'")
                
        elif key == ord('f'):  # F - Français
            language = 'fr'
            print("🇫🇷 Langue: Français")
            
        elif key == ord('a'):  # A - Arabe
            language = 'ar' 
            print("🇸🇦 اللغة: العربية")
            
        elif key == ord('e'):  # E - English
            language = 'en'
            print("🇺🇸 Language: English")
            
        elif key == ord('t'):  # T - Basculer mode
            translation_mode = not translation_mode
            mode_name = "TRADUCTION" if translation_mode else "LETTRES"
            print(f"🔁 Mode: {mode_name}")
            
        elif key == ord('q'):  # Q - Quitter
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 Session terminée!")
    print(f"📝 Mot final: '{current_word}'")
    if translation_mode and current_word:
        final_translation = smart_translation(current_word, language, True)
        print(f"🌍 Traduction: '{final_translation}'")

if __name__ == '__main__':
    main()
