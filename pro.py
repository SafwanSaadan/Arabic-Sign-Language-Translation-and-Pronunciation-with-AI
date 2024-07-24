import cv2
import numpy as np
import threading
import pyttsx3
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from googletrans import Translator
import os

class GUI:
    def __init__(self, root):
        # إعداد واجهة المستخدم الرسومية
        self.root = root
        self.root.title("تطبيق التعرف على لغة الإشارة العربية ونطق معناها")
        self.root.geometry("1370x710")  # تحديد حجم الواجهة
        self.root.resizable(width=False, height=False)  # تعيين قابلية التكبير والتصغير إلى القيمة False

        # إضافة صورة خلفية
        self.background_image = Image.open("AI.jpg")  # قم بتغيير اسم الصورة إلى اسم الصورة الخاصة بك
        self.background_image = self.background_image.resize((1370, 710), Image.BICUBIC)
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.background_label = tk.Label(root, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)


        # الحصول على حجم الشاشة
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # حساب موقع النافذة المركزية
        x_position = (screen_width - 1370) // 2
        y_position = 0

        # تحديد موقع النافذة المركزية
        self.root.geometry(f"1370x710+{x_position}+{y_position}")

        # تحميل النموذج والتسميات
        self.model = load_model("Holistic_keypoints_BiLSTM_model_3.h5", compile=False)
        self.class_names = open("labels2.txt", "r", encoding="utf-8").readlines()

        # إنشاء مكان لعرض صورة الكاميرا
        self.camera_frame = tk.Label(root, bg="#3498db", width=700, height=500)
        self.camera_frame.pack(pady=7)

        # إنشاء مكان لعرض النص
        self.class_label = tk.Label(root, text="", font=("Helvetica", 18, "bold"), bg="#3498db", fg="white", wraplength=500)
        self.class_label.pack(pady=7)

        # إنشاء مكان لعرض زر التقاط صورة وتبديل الكاميرا  
        captur_toggle_buttons = tk.Label(root)
        captur_toggle_buttons.pack(pady=2)

        # إضافة زر تبديل كاميرا الويب
        toggle_webcam_button = tk.Button(captur_toggle_buttons, text="تبديل كاميرا الويب", font=("Helvetica", 15), command=self.toggle_webcam, width=25, height=1, bg="#3498db", fg="white")
        toggle_webcam_button.pack(side="left", padx=25, pady=5)

        # زر لتحميل الصورة أو الفيديو
        load_button = tk.Button(captur_toggle_buttons, text="تحميل فيديو", font=("Helvetica", 15), command=self.load_file, width=30, height=1, bg="#2ecc71", fg="white")
        load_button.pack(side="right", padx=25, pady=5)

        # إضافة زر لتشغيل/إيقاف تشغيل الكاميرا
        self.toggle_camera_button = tk.Button(root, text="إيقاف الكاميرا", font=("Helvetica", 15), command=self.toggle_camera, width=30, height=1, bg="#e74c3c", fg="white")
        self.toggle_camera_button.pack(pady=5)

        # شريط في أسفل الواجهة (Footer)
        footer_label = tk.Label(root, text="تم التطوير بواسطة صفوان سعدان & حسام أحمد & عمار الشرعبي & أحمد الحلقبي", font=("Helvetica", 15), bg="#2c3e50", fg="white")
        footer_label.pack(side="bottom", fill="x")

        # متغير لتحديد ما إذا كان يجب عرض كاميرا الويب أو الفيديو
        self.camera_running = True
        self.video_running = False
        # إضافة متغير لتحديد ما إذا كان يجب عرض كاميرا الويب أم الصورة
        self.show_webcam = True

        # بدء تشغيل الكاميرا
        self.camera_thread = threading.Thread(target=self.update_camera)
        self.camera_thread.daemon = True
        self.camera_thread.start()

        # إعداد Mediapipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # متغير لتخزين مسار الملف
        self.file_path = None

    def toggle_camera(self):
        # تغيير حالة تشغيل/إيقاف تشغيل الكاميرا
        self.camera_running = not self.camera_running

        if self.camera_running:
            self.toggle_camera_button.config(text="إيقاف الكاميرا", bg="#e74c3c")
        else:
            self.toggle_camera_button.config(text="تشغيل الكاميرا", bg="#2ecc71")

    def load_file(self):
        # فتح نافذة لاختيار الفيديو
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])

        if file_path:
            self.file_path = file_path
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".mp4", ".avi"]:
                self.toggle_camera()  # إيقاف تشغيل الكاميرا عند تشغيل الفيديو
                self.load_video(file_path)

    def load_video(self, file_path):
        # إيقاف تشغيل الكاميرا
        self.camera_running = False
        self.video_running = True

        # فتح الفيديو وعرضه إطارًا تلو الآخر
        cap = cv2.VideoCapture(file_path)

        while cap.isOpened() and self.video_running:
            ret, frame = cap.read()
            if not ret:
                break

            # عكس الصورة أفقيًا
            frame = cv2.flip(frame, 1)
            # عملية الكشف والنقاط المفتاحية
            image, results = self.mediapipe_detection(frame, self.holistic)
            pose, lh, rh = self.extract_keypoints(results)
            keypoints = np.concatenate([pose, lh, rh])
            keypoints = keypoints.reshape(1, 48, 225, 3)  # تعديل الأبعاد لتتناسب مع نموذج LSTM

            # تنبؤ بالنموذج
            prediction = self.model.predict(keypoints)
            index = np.argmax(prediction)
            class_name = self.class_names[index].strip()
            confidence_score = prediction[0][index]
            class_text = f"{class_name}, بنسبة: {np.round(confidence_score * 100, 2)}%"

            # تحديث التسمية في واجهة المستخدم
            self.class_label.config(text=class_name)
            # self.speak(class_name)

            # عرض إطار الفيديو
            frame = cv2.resize(frame, (700, 500))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(frame)

            # تحديث إطار الفيديو في واجهة المستخدم
            self.camera_frame.config(image=photo)
            self.camera_frame.image = photo

            self.root.update_idletasks()
            self.root.update()

        cap.release()
        self.video_running = False

    def toggle_webcam(self):
        # تبديل بين كاميرا الويب والصورة الملتقطة
        self.show_webcam = not self.show_webcam

    def analyze_image(self, keypoints_array):
        # قم بتعديل مصفوفة النقاط لتتناسب مع شكل إدخال النموذج
        # على افتراض أن keypoints_array هي مصفوفة من النقاط المفتاحية لجميع الإطارات
        keypoints_array = np.array(keypoints_array, dtype=np.float32)
        keypoints_array = keypoints_array.reshape(1, 48, 225)  # تعديل الأبعاد لتتناسب مع نموذج LSTM

        # تنبؤ بالنموذج
        prediction = self.model.predict(keypoints_array)
        index = np.argmax(prediction)
        class_name = self.class_names[index].strip()
        confidence_score = prediction[0][index]

        # طباعة التنبؤ ونسبة الثقة
        print("Class:", class_name, end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        class_text = f"{class_name}, بنسبة: {np.round(confidence_score * 100, 2)}%"

        # تحديث التسمية في واجهة المستخدم
        self.class_label.config(text=class_name)

        # نص إلى كلام
        self.speak(class_name)

    def update_camera(self):
        camera = cv2.VideoCapture(0)

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Hand Detector
        detectorHand = HandDetector(detectionCon=0.9, maxHands=2)

        # قائمة لتخزين النقاط المفتاحية لكل إطار
        all_keypoints = []

        while True:
            if self.camera_running:
                if self.show_webcam:
                    webcam_source = 0
                else:
                    webcam_source = 1

                # camera = cv2.VideoCapture(webcam_source)

                # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                ret, frame = camera.read()

                # تعرف على الوجه باستخدام Haar Cascades
                if self.show_webcam:
                    faces = self.detect_faces(frame)
                    self.draw_faces(frame, faces)

                if not ret:
                    break

                # عكس الصورة أفقيًا
                frame = cv2.flip(frame, 1)
                # Find the hand and its landmarks
                hands, frame = detectorHand.findHands(frame, flipType=True)  # with draw

                # عملية الكشف والنقاط المفتاحية
                image, results = self.mediapipe_detection(frame, self.holistic)
                pose, lh, rh = self.extract_keypoints(results)
                keypoints = np.concatenate([pose, lh, rh])
                all_keypoints.append(keypoints)

                # عرض إطار كاميرا الويب
                image = cv2.resize(frame, (700, 500))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                photo = ImageTk.PhotoImage(image)

                # تحديث إطار الكاميرا في واجهة المستخدم
                self.camera_frame.config(image=photo)
                self.camera_frame.image = photo

                if len(all_keypoints) == 48:  # إذا كانت لدينا 48 إطارًا من النقاط المفتاحية
                    # تحليل الإطارات المجمعة
                    self.analyze_image(np.array(all_keypoints))
                    all_keypoints = []  # إعادة تعيين قائمة النقاط بعد التحليل

            self.root.update_idletasks()
            self.root.update()

        camera.release()


    def detect_faces(self, frame):
        # استخدام Haar Cascades للكشف عن الوجوه
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        return faces

    def draw_faces(self, frame, faces):
        # رسم مربعات حول الوجوه المكتشفة
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    def mediapipe_detection(self, image, model):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.process(image_rgb)
        return image, results

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        nose = pose[:3]
        lh_wrist = lh[:3]
        rh_wrist = rh[:3]
        pose_adjusted = self.adjust_landmarks(pose, nose)
        lh_adjusted = self.adjust_landmarks(lh, lh_wrist)
        rh_adjusted = self.adjust_landmarks(rh, rh_wrist)
        return pose_adjusted, lh_adjusted, rh_adjusted

    def adjust_landmarks(self, arr, center):
        arr_reshaped = arr.reshape(-1, 3)
        center_repeated = np.tile(center, (len(arr_reshaped), 1))
        arr_adjusted = arr_reshaped - center_repeated
        arr_adjusted = arr_adjusted.reshape(-1)
        return arr_adjusted

    def speak(self, text):
        # تهيئة المترجم
        translator = Translator()
        # ترجمة النص إلى الإنجليزية
        english_text = translator.translate(text, dest='en').text

        # تهيئة محرك TTS
        engine = pyttsx3.init()
        # نطق النص المترجم
        engine.say(english_text)
        # انتظر انتهاء التحدث
        engine.runAndWait()

# إنشاء الجذر (root) وبدء التطبيق
root = tk.Tk()
app = GUI(root)
root.mainloop()
