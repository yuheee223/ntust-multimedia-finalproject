import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from deepface import DeepFace

import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms as T, models
import mediapipe as mp

import tensorflow as tf

from keras.api.models import Sequential
from keras.api.layers import Rescaling
# from keras.layers.experimental.preprocessing import Rescaling
from keras.api.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.api.layers import BatchNormalization
from keras.api.losses import categorical_crossentropy
from keras.api.optimizers import Adam


# for情緒識別
class VGGNet(Sequential):
    def __init__(self, input_shape, num_classes, checkpoint_path, lr=1e-3):
        super().__init__()
        self.add(Rescaling(1./255, input_shape=input_shape))
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Flatten())
        
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(256, activation='relu'))

        self.add(Dense(num_classes, activation='softmax'))

        self.compile(optimizer=Adam(learning_rate=lr),
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])
        
        self.checkpoint_path = checkpoint_path

# 圖片處理
class ImageProcessor:
    def __init__(self):
        """人像分割"""
        # 載入預訓練模型-DeepLabv3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 若電腦支援CUDA硬體且安裝正確版本的PyTorch則啟用GPU以加速處理速度
        self.model = deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1).to(self.device)
        self.model.eval()
        
        # 模型正規化參數
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # 模型轉換
        self.transform = T.Compose([
            T.Resize(512),  # 固定輸入大小
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        """情緒識別"""
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotions_info = {# 框      字            背景
            'None': [(255,255,255), (0,0,0), (255,255,255)],
            'Angry': [(0,0,255), (255,255,255), (255, 105, 97)],         # 紅色
            'Disgust': [(0,102,0), (255,255,255), (255,255,255)],
            'Fear': [(255,255,153), (0,51,51), (255,255,255)],
            'Happy': [(153,0,153), (255,255,255), (255, 223, 186)],      # 橘色
            'Sad': [(255,0,0), (255,255,255), (135, 206, 235)],          # 藍色
            'Surprise': [(0,255,0), (255,255,255), (255,255,255)],
            'Neutral': [(160,160,160), (255,255,255), (211, 211, 211)]   # 灰色
        }
        num_classes = len(self.emotions)
        input_shape = (48, 48, 1)
        weights_1 = 'model/vggnet.h5'
        weights_2 = 'model/vggnet_up.h5'

        self.model_1 = VGGNet(input_shape, num_classes, weights_1)
        self.model_1.load_weights(self.model_1.checkpoint_path)

        self.model_2 = VGGNet(input_shape, num_classes, weights_2)
        self.model_2.load_weights(self.model_2.checkpoint_path)

        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils     # 畫出特徵點
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        """人臉特效"""
        # 初始化 MediaPipe Face Mesh 模組
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    """for情緒識別"""
    def detection_preprocessing(self, image, w_max=500):
        H, W, _ = image.shape
        if W > w_max:
            ratio = w_max / W
            h = int(H * ratio)
            image = cv2.resize(image, (w_max, h))
        return image
    def resize_face(self, face):
        x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
        return tf.image.resize(x, (48,48))
    def recognition_preprocessing(self, faces):
        x = tf.convert_to_tensor([self.resize_face(f) for f in faces])
        return x

    """找到圖片中的最大臉並取得其情緒"""
    def get_biggestface_and_emotion(self, image):
        H, W, _ = image.shape
        
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))   # 將image轉換為RGB格式
        if results.detections:
            faces = []
            pos = []
            biggest_box = 0
            for i, detection in enumerate(results.detections):
                box = detection.location_data.relative_bounding_box
                # mp_drawing.draw_detection(image, detection)

                x = int(box.xmin * W)
                y = int(box.ymin * H)
                w = int(box.width * W)
                h = int(box.height * H)

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, W)
                y2 = min(y + h, H)

                face = image[y1:y2,x1:x2]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                faces.append(face)
                pos.append((x1, y1, x2, y2))

                if w*h > biggest_box:
                    biggest_i = i
        
            x = self.recognition_preprocessing(faces)

            y_1 = self.model_1.predict(x)
            y_2 = self.model_2.predict(x)
            l = np.argmax(y_1+y_2, axis=1)
            # emotion = self.emotions[l[biggest_i]]
            print(pos[biggest_i], self.emotions[l[biggest_i]])

            # 創建遮罩 (與原圖大小相同)
            biggest_face_mask = np.zeros((H, W), dtype=np.uint8)
            # 取得最大臉的四點
            x1, y1, x2, y2 = pos[biggest_i]
            # 增加邊界和確保分辨率
            padding = 0.1
            x1 = max(0, int(x1 - padding * W))
            y1 = max(0, int(y1 - padding * H))
            x2 = min(W, int(x2 + padding * W))
            y2 = min(H, int(y2 + padding * H))
            # 在遮罩上將臉部區域設為白色
            biggest_face_mask[y1:y2, x1:x2] = 255

            return biggest_face_mask, self.emotions[l[biggest_i]]

            # for i in range(len(faces)):
            #     if i == biggest_i:
            #         cv2.rectangle(image, (pos[i][0],pos[i][1]),
            #                         (pos[i][2],pos[i][3]), self.emotions[l[i]][1], 2, lineType=cv2.LINE_AA)                 
            #         cv2.rectangle(image, (pos[i][0],pos[i][1]-20),
            #                         (pos[i][2]+20,pos[i][1]), self.emotions[l[i]][1], -1, lineType=cv2.LINE_AA)                    
            #         cv2.putText(image, f'{self.emotions[l[i]][0]}', (pos[i][0],pos[i][1]-5),
            #                         0, 0.6, self.emotions[l[i]][2], 2, lineType=cv2.LINE_AA)
            #         # 調整背景
            #     else:
            #         cv2.rectangle(image, (pos[i][0],pos[i][1]),
            #                         (pos[i][2],pos[i][3]), (255,255,255), 2, lineType=cv2.LINE_AA)
                    
            #         cv2.rectangle(image, (pos[i][0],pos[i][1]-20),
            #                         (pos[i][2]+20,pos[i][1]), (255,255,255), -1, lineType=cv2.LINE_AA)
                    
            #         cv2.putText(image, f'{self.emotions[l[i]][0]}', (pos[i][0],pos[i][1]-5),
            #                         0, 0.6, (0,0,0), 2, lineType=cv2.LINE_AA)
        # return image
        return np.zeros((H, W), dtype=np.uint8), "None" # 找不到人臉

    """分割圖片並返回最大區域遮罩"""
    def segment_image(self, image):
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        # 模型預測
        with torch.no_grad():
            output = self.model(image_tensor)['out']
            output_predictions = F.softmax(output, dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
        # 提取人像遮罩
        mask = np.uint8(output_predictions == 15) * 255
        return mask

    """找到最大連通區域"""
    def get_largest_connected_area(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 忽略背景 (index 0)
        largest_mask = np.uint8(labels == max_label) * 255      # 保留最大區域
        return largest_mask

    """調整圖片色調"""
    @staticmethod
    def change_backgroung_color(image, target_color):
        # 將圖片從 BGR 轉換為 HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  

        """調整色相 (Hue)"""
        # 獲取目標顏色的 H 值 (色相)
        target_hue = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)[0][0][0]   
        # 計算當前圖片的色相
        current_hue = np.mean(hsv_image[:, :, 0])   
        # 計算色相偏移
        hue_shift = target_hue - current_hue
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180     # 保持色相範圍在 [0, 180]

        """調整亮度 (Value)"""
        brightness_factor = 0.7 # <1.0降低亮度；>1.0提高亮
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness_factor, 0, 255).astype(np.uint8)

        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    """""""""""""effect func"""""""""""""

    """Sad 眼淚特效"""
    @staticmethod
    def add_tear_effect(image, face_landmarks):
        H, W, _ = image.shape

        left_eye_center = face_landmarks.landmark[33]   # 左眼中央
        right_eye_center = face_landmarks.landmark[263] # 右眼中央

        # 計算眼睛位置，這裡將眼睛位置下方畫圓表示眼淚
        left_x = int(left_eye_center.x * W)
        left_y = int(left_eye_center.y * H)

        right_x = int(right_eye_center.x * W)
        right_y = int(right_eye_center.y * H)

        # 水滴的寬度和高度
        width = 10  # 水滴的寬度
        height = 40  # 水滴的高度

        # 1. 畫出三角形（頂部為水滴的尖端）
        triangle_points = np.array([[
            (left_x, left_y+20),  # 頂部
            (left_x - width, left_y + height),  # 左側底角
            (left_x + width, left_y + height)  # 右側底角
        ]], dtype=np.int32)

        # 畫三角形
        cv2.fillPoly(image, triangle_points, color=(0,55,174)) # HEX #0037ae
        # cv2.polylines(final_image, triangle_points, isClosed=True, color=(0, 0, 255), thickness=2)

        # 2. 畫出半圓形（底部）
        cv2.ellipse(image, 
                    (left_x, left_y + height),  # 半圓心
                    (width, height // 2),       # 半圓的長短軸
                    0,                          # 旋轉角度
                    0,                          # 起始角度
                    180,                        # 結束角度
                    (0,55,174),                # 顏色：藍色
                    -1)                         # 填充顏色
        
        transition_radius = 5  # 過渡圓形的半徑
        transition_y = left_y + height - transition_radius  # 確保圓形位於兩個形狀之間的接縫處
        cv2.circle(image, (left_x, transition_y), transition_radius, (0,55,174), -1)  # 過渡圓形

        # 1. 畫出三角形（頂部為水滴的尖端）
        triangle_points = np.array([[
            (right_x, right_y+20),  # 頂部
            (right_x - width, right_y + height),  # 左側底角
            (right_x + width, right_y + height)  # 右側底角
        ]], dtype=np.int32)

        # 畫三角形
        cv2.fillPoly(image, triangle_points, color=(0,55,174))
        # cv2.polylines(final_image, triangle_points, isClosed=True, color=(0, 0, 255), thickness=2)

        # 2. 畫出半圓形（底部）
        cv2.ellipse(image, 
                    (right_x, right_y + height),  # 半圓心
                    (width, height // 2),       # 半圓的長短軸
                    0,                          # 旋轉角度
                    0,                          # 起始角度
                    180,                        # 結束角度
                    (0,55,174),                # 顏色：藍色
                    -1)                         # 填充顏色
        
        transition_y = right_y + height - transition_radius  # 確保圓形位於兩個形狀之間的接縫處
        cv2.circle(image, (right_x, transition_y), transition_radius, (0,55,174), -1)  # 過渡圓形


        return image
 
    """ Happy 彩虹特效"""
    @staticmethod
    def add_rainbow_effect(image, face_landmarks):
        H, W, _ = image.shape

        # 頭頂位置
        top_x = int(face_landmarks.landmark[10].x * W)
        top_y = int(face_landmarks.landmark[10].y * H)

        # 計算鼻尖與頭頂的相對座標
        nose_x = int(face_landmarks.landmark[1].x * W)  
        nose_y = int(face_landmarks.landmark[1].y * H)

        # 偏航角（Yaw）: 從鼻子與頭頂的水平方向上的差異計算
        delta_y = top_y - nose_y
        delta_x = top_x - nose_x
        yaw_angle = np.arctan2(delta_y, delta_x) * 180 / np.pi  # 轉換為角度
        print(yaw_angle)
    
        left_eye_center = face_landmarks.landmark[33]   # 左眼中央
        right_eye_center = face_landmarks.landmark[263] # 右眼中央
        left_x = int(left_eye_center.x * W)
        right_x = int(right_eye_center.x * W)

        # 在頭頂上方畫彩虹
        face_width = abs(right_x - left_x)
        rainbow_radius = int(face_width * 1.5) # 彩虹的半徑
        rainbow_height = int(rainbow_radius*0.1)  # 彩虹的間隔高度
        rainbow_colors = [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (75, 0, 130), (148, 0, 211)]  # 紅、橙、黃、綠、藍、靛、紫

        # rainbow_colors = [
        #     (255, 0, 0, 200),   # 紅色，半透明
        #     (255, 127, 0, 200), # 橙色，半透明
        #     (255, 255, 0, 200), # 黃色，半透明
        #     (0, 255, 0, 200),   # 綠色，半透明
        #     (0, 0, 255, 200),   # 藍色，半透明
        #     (75, 0, 130, 200),  # 靛色，半透明
        #     (148, 0, 211, 200)  # 紫色，半透明
        # ]

        # # 創建一個全透明的圖層來繪製彩虹
        # rainbow_layer = np.zeros_like(image)  # 全透明圖層，與 final_image 相同的大小


        # 繪製彩虹的半圓
        for i, color in enumerate(rainbow_colors):
            # 半圓的起始和結束角度
            angle_start = 180
            angle_end = 360

            # 計算每個半圓的半徑（每個顏色的半圓逐漸變小）
            radius = rainbow_radius - i * rainbow_height
            
            # 調整圓形的繪製方式，使其顏色從上至下
            cv2.ellipse(image, 
                        (top_x, top_y - 40),  # 半圓的中心（頂部稍微向上偏移）
                        (radius, radius),         # 半徑
                        yaw_angle+90,                 # 旋轉角度
                        angle_start,               # 起始角度
                        angle_end,                 # 結束角度
                        color,                     # 顏色
                        -1)                        # 填充顏色
            
        # 合併原圖和彩虹圖層
        # final_image_with_rainbow = cv2.add(image, rainbow_layer)
        # return final_image_with_rainbow

        return image


# GUI
class GUI:
    def __init__(self, master):
        self.master = master
        master.title("MultiMedia")
        master.state('zoomed')

        self.processor = ImageProcessor()

        """處理前"""
        self.frame_before = tk.Frame(master, background="#333")
        self.frame_before.pack(fill='both', side='left', expand=1)

        self.open_img_btn = tk.Button(self.frame_before, text="Open Image", command=self.open_img)
        self.open_img_btn.pack()

        self.imgLabel_before = tk.Label(self.frame_before)
        self.imgLabel_before.pack()

        """處理後"""
        self.frame_after = tk.Frame(master, background="#999")
        self.frame_after.pack(fill='both', side='right', expand=1)

        self.imgLabel_after = tk.Label(self.frame_after, text="Processed Image")
        self.imgLabel_after.pack()

    """開啟圖片"""
    def open_img(self):
        file_types = [('Image Files', '*.png;*.jpg;*.jpeg')]
        path = filedialog.askopenfilename(filetypes=file_types)
        if not path:
            print("No file chosen!")
            return

        image = Image.open(path)
        self.display_image(image, self.imgLabel_before)
        self.process_img(image)

    """顯示圖片"""
    def display_image(self, image, label):
        width = 500
        height = int(image.height * width / image.width)
        image = image.resize((width, height))
        image = ImageTk.PhotoImage(image)
        label.configure(image=image)
        label.image = image

    # 處理圖片
    def process_img(self, image):
        """分割圖片"""
        original_image = np.array(image)    # 原始圖片作為NumPy陣列
        people_mask = self.processor.segment_image(image)
        people_mask = cv2.resize(people_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)    # 確保遮罩和原始圖片大小一致。將遮罩從模型的輸出大小（T.Resize）縮放到與原始圖片一致的大小
        largest_people_mask = self.processor.get_largest_connected_area(people_mask)

        """提取人像"""
        people_image = np.where(largest_people_mask[..., None] == 255, original_image, 0) 
        # people_image = cv2.bitwise_and(original_image, original_image, mask=largest_people_mask)  # 效果一樣
        # Image.fromarray(people_image).show()

        """取得最大臉、並識別情緒"""
        biggest_face_mask, emotion = self.processor.get_biggestface_and_emotion(people_image)
        biggest_face_image = np.where(biggest_face_mask[..., None] == 255, original_image, 0)   # 原圖片中只顯示最大臉，以在後續只針對最大臉作人臉特徵點
        # Image.fromarray(biggest_face_image).show()

        """檢測人臉特徵點"""
        results = self.processor.face_mesh.process(cv2.cvtColor(biggest_face_image, cv2.COLOR_BGR2RGB))  # 將people_image轉換為RGB格式以適配MediaPipe
        # try:
        if results.multi_face_landmarks:    # 檢查是否檢測到人臉特徵點
            # for face_landmarks in results.multi_face_landmarks:
            #     for landmark in face_landmarks.landmark:
            #         # 計算特徵點在圖片中的座標
            #         x = int(landmark.x * people_image.shape[1])
            #         y = int(landmark.y * people_image.shape[0])
            #         # 在每個五官的特徵點上繪製圓點
            #         cv2.circle(people_image, (x, y), 1, (0, 255, 0), -1)  # 綠色圓點，大小為 1         

            """取得背景遮罩以針對情緒變更背景顏色"""
            background_mask = np.uint8(largest_people_mask == 0) * 255  # 反轉people_mask遮罩
            background_image = np.where(background_mask[..., None] == 255, original_image, 0)   # 提取背景
            background_image = self.processor.change_backgroung_color(background_image, self.processor.emotions_info[emotion][2])    # 變更背景顏色

            """結合人像及背景"""
            final_image = people_image + background_image

            """根據情緒加上對應特效"""
            face_landmarks = results.multi_face_landmarks[0]    # 取得人臉特徵點
            if emotion == 'Sad':
                final_image = self.processor.add_tear_effect(final_image, face_landmarks)
            elif emotion == 'Happy':
                final_image = self.processor.add_rainbow_effect(final_image, face_landmarks)

            """顯示處理後圖片"""
            final_image_pil = Image.fromarray(final_image)
            self.display_image(final_image_pil, self.imgLabel_after)
        # except:
        else:
            """沒有最大臉(biggest_face_image全黑, emotion=="None") 或 沒檢測到人臉特徵點(最大臉像素太小)"""
            print("No facial landmarks detected")
            self.imgLabel_after.configure(image="")
            self.imgLabel_after.image = None
            self.imgLabel_after.configure(text="Sorry, the face is too small.")


# 主程式
if __name__ == "__main__":
    app = tk.Tk()
    gui = GUI(app)
    app.mainloop()