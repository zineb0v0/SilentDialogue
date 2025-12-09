import sys
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                             QFrame, QButtonGroup, QRadioButton, QProgressBar)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from utils import load_trained_model, load_labels, preprocess_image_bgr
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = 'models/asl_model_latest.h5'
PAD = 70

SEMANTIC_TRANSLATIONS = {
    'en': {
        'I': 'I', 'YOU': 'You', 'HE': 'He', 'SHE': 'She', 'WE': 'We', 'THEY': 'They',
        'ME': 'Me', 'MY': 'My', 'YOUR': 'Your', 'OUR': 'Our',
        'HELLO': 'Hello', 'HI': 'Hi', 'GOODBYE': 'Goodbye', 'BYE': 'Bye',
        'THANKS': 'Thanks', 'THANKYOU': 'Thank you', 'PLEASE': 'Please',
        'EXCUSEME': 'Excuse me', 'SORRY': 'Sorry', 'STOP': 'Stop', 'GO': 'Go',
        'WHAT': 'What', 'WHERE': 'Where', 'WHEN': 'When', 'WHY': 'Why',
        'HOW': 'How', 'WHO': 'Who',
        'HAPPY': 'Happy', 'SAD': 'Sad', 'ANGRY': 'Angry', 'LOVE': 'Love', 'LIKE': 'Like',
        'FAMILY': 'Family', 'FRIEND': 'Friend', 'HELP': 'Help',
        'FOOD': 'Food', 'WATER': 'Water', 'SLEEP': 'Sleep', 'WORK': 'Work',
        'SCHOOL': 'School', 'HOME': 'Home', 'MONEY': 'Money', 'DOG': 'Dog', 'CAT': 'Cat',
        'AM': 'am', 'IS': 'is', 'ARE': 'are', 'WAS': 'was', 'WERE': 'were', 'BE': 'be',
        'HAVE': 'have', 'HAS': 'has', 'DO': 'do', 'DOES': 'does', 'GOING': 'going',
        'THIS': 'this', 'MY': 'my', 'PROJECT': 'project', 'GOOD': 'good', 'BAD': 'bad',
        'TODAY': 'today', 'TOMORROW': 'tomorrow', 'YESTERDAY': 'yesterday', 'NOW': 'now',
    },
    'fr': {
        'I': 'Je', 'YOU': 'Tu', 'HE': 'Il', 'SHE': 'Elle', 'WE': 'Nous', 'THEY': 'Ils',
        'ME': 'Moi', 'MY': 'Mon', 'YOUR': 'Ton', 'OUR': 'Notre',
        'HELLO': 'Bonjour', 'HI': 'Salut', 'GOODBYE': 'Au revoir', 'BYE': 'Salut',
        'THANKS': 'Merci', 'THANKYOU': 'Merci', 'PLEASE': 'S il vous plaît',
        'EXCUSEME': 'Excusez-moi', 'SORRY': 'Désolé', 'STOP': 'Arrête', 'GO': 'Aller',
        'WHAT': 'Quoi', 'WHERE': 'Où', 'WHEN': 'Quand', 'WHY': 'Pourquoi',
        'HOW': 'Comment', 'WHO': 'Qui',
        'HAPPY': 'Heureux', 'SAD': 'Triste', 'ANGRY': 'En colère', 'LOVE': 'Aimer',
        'FAMILY': 'Famille', 'FRIEND': 'Ami', 'HELP': 'Aider',
        'FOOD': 'Nourriture', 'WATER': 'Eau', 'SLEEP': 'Dormir', 'WORK': 'Travail',
        'THIS': 'ce', 'MY': 'mon', 'PROJECT': 'projet', 'GOOD': 'bon', 'BAD': 'mauvais',
    },
    'ar': {
        'I': 'أنا', 'YOU': 'أنت', 'HE': 'هو', 'SHE': 'هي', 'WE': 'نحن', 'THEY': 'هم',
        'HELLO': 'مرحبا', 'HI': 'أهلا', 'GOODBYE': 'مع السلامة', 'BYE': 'وداعا',
        'THANKS': 'شكرا', 'THANKYOU': 'شكرا لك', 'PLEASE': 'من فضلك',
        'WHAT': 'ماذا', 'WHERE': 'أين', 'WHEN': 'متى', 'WHY': 'لماذا',
        'HAPPY': 'سعيد', 'SAD': 'حزين', 'ANGRY': 'غاضب', 'LOVE': 'يحب',
        'FAMILY': 'عائلة', 'FRIEND': 'صديق', 'HELP': 'يساعد',
        'FOOD': 'طعام', 'WATER': 'ماء', 'WORK': 'عمل', 'HOME': 'بيت',
    }
}

def detect_semantic_words(text):
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

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, object, str, float)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.model = load_trained_model(MODEL_PATH)
        self.labels = load_labels(models_dir='models')
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.label_buffer = []
        self.buffer_len = 8
        self.confidence_threshold = 0.7
        self.cooldown_frames = 0
        
    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                
                if self.cooldown_frames > 0:
                    self.cooldown_frames -= 1
                
                current_prediction = None
                current_confidence = 0
                hand_landmarks = None
                
                if results.multi_hand_landmarks and self.cooldown_frames == 0:
                    lm = results.multi_hand_landmarks[0]
                    hand_landmarks = lm
                    
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
                        preds = self.model.predict(x, verbose=0)
                        idx = int(np.argmax(preds))
                        prob = float(np.max(preds))
                        label = self.labels[idx]
                        
                        if prob > self.confidence_threshold:
                            self.label_buffer.append((label, prob))
                            if len(self.label_buffer) > self.buffer_len:
                                self.label_buffer.pop(0)
                            if self.label_buffer:
                                votes = {}
                                for L, P in self.label_buffer:
                                    votes[L] = votes.get(L, 0) + (P * 10)
                                current_prediction = max(votes.items(), key=lambda x: x[1])[0]
                                current_confidence = max(p for l, p in self.label_buffer if l == current_prediction)
                
                # Dessiner les landmarks
                if hand_landmarks:
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(140, 20, 252), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(200, 150, 255), thickness=2)
                    )
                
                self.change_pixmap_signal.emit(frame, hand_landmarks, 
                                               current_prediction if current_prediction else "", 
                                               current_confidence)
        
        cap.release()
    
    def stop(self):
        self.running = False
        self.wait()

class ModernCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #323238;
                border: 2px solid #C896FF;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout()
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #787882;
                font-size: 11px;
                font-weight: bold;
                border: none;
                padding: 0px;
            }
        """)
        
        self.content_label = QLabel("...")
        self.content_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
                border: none;
                padding: 10px 0px;
            }
        """)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.content_label)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def set_content(self, text):
        self.content_label.setText(text if text else "...")

class ASLInterpreter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Silent Dialogue")
        self.setGeometry(100, 100, 1200, 700)
        
        # Variables
        self.current_word = ""
        self.language = 'en'
        self.translation_mode = True
        self.auto_add_mode = False
        self.stable_frames = 0
        self.required_stable_frames = 15
        self.last_added_prediction = None
        self.saved_texts = []
        
        # Style global
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0F0F14;
            }
            QPushButton {
                background-color: #323238;
                color: white;
                border: 2px solid #787882;
                border-radius: 8px;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3F3F48;
                border: 2px solid #C896FF;
            }
            QPushButton:checked {
                background-color: #8C14FC;
                border: 2px solid #C896FF;
            }
            QLabel {
                color: white;
            }
            QTextEdit {
                background-color: #323238;
                color: white;
                border: 2px solid #C896FF;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }
            QProgressBar {
                border: 2px solid #C896FF;
                border-radius: 5px;
                text-align: center;
                background-color: #323238;
            }
            QProgressBar::chunk {
                background-color: #8C14FC;
                border-radius: 3px;
            }
        """)
        
        self.init_ui()
        
        # Thread vidéo
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_frame)
        self.video_thread.start()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # ============ LEFT PANEL: CAMERA ============
        left_panel = QVBoxLayout()
        
        # Titre
        title_label = QLabel("ASL SILENT DIALOGUE")
        title_label.setStyleSheet("""
            QLabel {
                color: #C896FF;
                font-size: 28px;
                font-weight: bold;
                padding: 10px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 3px solid #8C14FC;
                border-radius: 10px;
                background-color: black;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        
        left_panel.addWidget(title_label)
        left_panel.addWidget(self.video_label, alignment=Qt.AlignCenter)
        left_panel.addStretch()
        
        # ============ RIGHT PANEL: CONTROLS ============
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)
        
        # Detection Status Card
        self.status_card = QFrame()
        self.status_card.setStyleSheet("""
            QFrame {
                background-color: #323238;
                border: 2px solid #787882;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        status_layout = QVBoxLayout()
        
        self.status_title = QLabel("DETECTION STATUS")
        self.status_title.setStyleSheet("color: #787882; font-size: 11px; font-weight: bold;")
        
        self.prediction_label = QLabel("WAITING...")
        self.prediction_label.setStyleSheet("color: white; font-size: 22px; font-weight: bold;")
        
        self.confidence_label = QLabel("0% - N/A")
        self.confidence_label.setStyleSheet("color: #C896FF; font-size: 12px;")
        
        status_layout.addWidget(self.status_title)
        status_layout.addWidget(self.prediction_label)
        status_layout.addWidget(self.confidence_label)
        self.status_card.setLayout(status_layout)
        
        # Letters Card
        self.letters_card = ModernCard("LETTERS")
        
        # Translation Card
        self.translation_card = ModernCard("TRANSLATION")
        
        # Auto-Add Button with Progress
        auto_layout = QVBoxLayout()
        self.auto_add_btn = QPushButton("⚡ AUTO-ADD: OFF")
        self.auto_add_btn.setCheckable(True)
        self.auto_add_btn.clicked.connect(self.toggle_auto_add)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.required_stable_frames)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(5)
        
        auto_layout.addWidget(self.auto_add_btn)
        auto_layout.addWidget(self.progress_bar)
        
        # Translation Mode Button
        self.trans_mode_btn = QPushButton("TRANSLATION MODE")
        self.trans_mode_btn.setCheckable(True)
        self.trans_mode_btn.setChecked(True)
        self.trans_mode_btn.clicked.connect(self.toggle_translation)
        
        # Language Buttons
        lang_layout = QHBoxLayout()
        lang_layout.setSpacing(10)
        
        self.lang_group = QButtonGroup()
        
        self.btn_en = QPushButton("🇬🇧 EN")
        self.btn_en.setCheckable(True)
        self.btn_en.setChecked(True)
        self.btn_en.clicked.connect(lambda: self.set_language('en'))
        
        self.btn_fr = QPushButton("🇫🇷 FR")
        self.btn_fr.setCheckable(True)
        self.btn_fr.clicked.connect(lambda: self.set_language('fr'))
        
        self.btn_ar = QPushButton("🇸🇦 AR")
        self.btn_ar.setCheckable(True)
        self.btn_ar.clicked.connect(lambda: self.set_language('ar'))
        
        self.lang_group.addButton(self.btn_en)
        self.lang_group.addButton(self.btn_fr)
        self.lang_group.addButton(self.btn_ar)
        
        lang_layout.addWidget(self.btn_en)
        lang_layout.addWidget(self.btn_fr)
        lang_layout.addWidget(self.btn_ar)
        
        # Action Buttons
        action_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("+ Add Letter")
        self.add_btn.clicked.connect(self.add_letter)
        
        self.delete_btn = QPushButton("⌫ Delete")
        self.delete_btn.clicked.connect(self.delete_letter)
        
        action_layout.addWidget(self.add_btn)
        action_layout.addWidget(self.delete_btn)
        
        # clear and Save Buttons

        util_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton(" Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.clicked.connect(self.save_text)
        
        util_layout.addWidget(self.clear_btn)
        util_layout.addWidget(self.save_btn)
        
        # Ajouter tous les widgets au panel droit
        right_panel.addWidget(self.status_card)
        right_panel.addWidget(self.letters_card)
        right_panel.addWidget(self.translation_card)
        right_panel.addLayout(auto_layout)
        right_panel.addWidget(self.trans_mode_btn)
        right_panel.addLayout(lang_layout)
        right_panel.addLayout(action_layout)
        right_panel.addLayout(util_layout)
        right_panel.addStretch()
        
        # Assembler les panels
        main_layout.addLayout(left_panel, 60)
        main_layout.addLayout(right_panel, 40)
        
        central_widget.setLayout(main_layout)
        
        # Variables pour la prédiction actuelle
        self.current_prediction = None
        self.current_confidence = 0
    
    def update_frame(self, frame, hand_landmarks, prediction, confidence):
        self.current_prediction = prediction
        self.current_confidence = confidence
        
        # Convertir frame pour Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
        # Mettre à jour le status
        if prediction and confidence > 0:
            self.prediction_label.setText(prediction.upper())
            
            if confidence > 0.85:
                status_text = "EXCELLENT"
                color = "#FAB1E2"
            elif confidence > 0.70:
                status_text = "GOOD"
                color = "#F08FD6"
            else:
                status_text = "LOW"
                color = "#787882"
            
            self.confidence_label.setText(f"{confidence*100:.0f}% - {status_text}")
            self.confidence_label.setStyleSheet(f"color: {color}; font-size: 12px;")
            self.status_card.setStyleSheet(f"""
                QFrame {{
                    background-color: #323238;
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 15px;
                }}
            """)
        else:
            self.prediction_label.setText("WAITING...")
            self.confidence_label.setText("0% - N/A")
            self.status_card.setStyleSheet("""
                QFrame {
                    background-color: #323238;
                    border: 2px solid #787882;
                    border-radius: 12px;
                    padding: 15px;
                }
            """)
        
        # Auto-add logic
        if self.auto_add_mode and prediction and prediction != 'nothing':
            if prediction == self.last_added_prediction:
                self.stable_frames += 1
            else:
                self.stable_frames = 0
                self.last_added_prediction = prediction
            
            self.progress_bar.setValue(self.stable_frames)
            
            if self.stable_frames >= self.required_stable_frames and confidence > 0.8:
                if prediction == 'space':
                    self.current_word += ' '
                elif prediction == 'del':
                    self.current_word = self.current_word[:-1] if self.current_word else ""
                else:
                    self.current_word += prediction
                
                self.stable_frames = 0
                self.last_added_prediction = None
                self.video_thread.cooldown_frames = 10
                self.update_displays()
        else:
            self.stable_frames = 0
            self.progress_bar.setValue(0)
    
    def update_displays(self):
        self.letters_card.set_content(self.current_word if self.current_word else "...")
        
        if self.translation_mode:
            translation = smart_translation(self.current_word, self.language, True)
            self.translation_card.set_content(translation if translation else "...")
        else:
            self.translation_card.set_content("Disabled")
    
    def add_letter(self):
        if self.current_prediction and self.current_prediction != 'nothing':
            if self.current_prediction == 'space':
                self.current_word += ' '
            elif self.current_prediction == 'del':
                self.current_word = self.current_word[:-1] if self.current_word else ""
            else:
                self.current_word += self.current_prediction
            
            self.video_thread.cooldown_frames = 20
            self.update_displays()
    
    def delete_letter(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.update_displays()
    
    def clear_all(self):
        self.current_word = ""
        self.update_displays()
    
    def toggle_auto_add(self):
        self.auto_add_mode = self.auto_add_btn.isChecked()
        self.auto_add_btn.setText(f"⚡ AUTO-ADD: {'ON' if self.auto_add_mode else 'OFF'}")
        self.stable_frames = 0
        self.last_added_prediction = None
    
    def toggle_translation(self):
        self.translation_mode = self.trans_mode_btn.isChecked()
        self.update_displays()
    
    def set_language(self, lang):
        self.language = lang
        print(f"Language changed to: {lang}")
        self.update_displays()
    
    def save_text(self):
        if self.current_word:
            translation = smart_translation(self.current_word, self.language, True)
            self.saved_texts.append({
                'text': self.current_word,
                'translation': translation,
                'language': self.language
            })
            print(f"💾 Saved: {self.current_word} → {translation}")
    
    def closeEvent(self, event):
        self.video_thread.stop()
        
        if self.saved_texts:
            print("\n📝 Saved Texts:")
            for i, item in enumerate(self.saved_texts, 1):
                print(f"  {i}. {item['text']} → {item['translation']} ({item['language']})")
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = ASLInterpreter()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()