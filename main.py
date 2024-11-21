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


# 初始化 MediaPipe Face Mesh 模組
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# 圖片處理
class ImageProcessor:
    def __init__(self):
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

    def segment_image(self, image):
        """分割圖片並返回最大區域遮罩"""
        original_image = np.array(image.convert("RGB"))
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 模型預測
        with torch.no_grad():
            output = self.model(image_tensor)['out']
            output_predictions = F.softmax(output, dim=1).argmax(dim=1).squeeze(0).cpu().numpy()

        # 提取人像遮罩
        mask = np.uint8(output_predictions == 15) * 255
        return original_image, mask

    def get_largest_connected_area(self, mask):
        """找到最大連通區域"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 忽略背景 (index 0)
        largest_mask = np.uint8(labels == max_label) * 255      # 保留最大區域
        return largest_mask

    def adjust_background(self, original_image, mask, emotion):
        """調整背景顏色"""
        background_color = self.get_background_color(emotion)
        background_mask = np.uint8(mask == 0) * 255  # 反轉遮罩
        background = cv2.bitwise_and(original_image, original_image, mask=background_mask)
        return self.change_hue(background, background_color)

    @staticmethod
    def get_background_color(emotion):
        """根據情緒選擇背景顏色"""
        emotion_to_color = {
            'happy': (255, 223, 186),   # 橘色
            'sad': (135, 206, 235),     # 藍色
            'angry': (255, 105, 97),    # 紅色
            'neutral': (211, 211, 211), # 灰色
        }
        return emotion_to_color.get(emotion, (255, 255, 255))   # 默認返回白色

    @staticmethod
    def change_hue(image, target_color):
        """調整圖片色調"""
        # 將圖片從 BGR 轉換為 HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  
        # 獲取目標顏色的 H 值 (色相)
        target_hue = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)[0][0][0]   
        # 計算當前圖片的色相
        current_hue = np.mean(hsv_image[:, :, 0])   
        # 計算色相偏移
        hue_shift = target_hue - current_hue
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180     # 保持色相範圍在 [0, 180]
        # 調整亮度 (Value)
        brightness_factor = 0.7 # <1.0降低亮度；>1.0提高亮
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness_factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    def emotion_recognize(self, image):
        """情緒分析"""
        try:
            emotion = DeepFace.analyze(image, actions=['emotion'])[0]['dominant_emotion']
            print(f"Emotion: {emotion}")
        except:
            print("Failed to analyze emotion")

    # @staticmethod
    # def add_rainbow_effect(image):
    #     """添加彩虹特效（示例）"""

    # @staticmethod
    # def add_tear_effect(image, face_landmarks):
 
# GUI
class GUI:
    def __init__(self, master):
        self.master = master
        master.title("MultiMedia")
        master.state('zoomed')

        self.processor = ImageProcessor()

        # 處理前
        self.frame_before = tk.Frame(master, background="#333")
        self.frame_before.pack(fill='both', side='left', expand=1)

        self.open_img_btn = tk.Button(self.frame_before, text="Open Image", command=self.open_img)
        self.open_img_btn.pack()

        self.imgLabel_before = tk.Label(self.frame_before)
        self.imgLabel_before.pack()

        # 處理後
        self.frame_after = tk.Frame(master, background="#999")
        self.frame_after.pack(fill='both', side='right', expand=1)

        self.imgLabel_after = tk.Label(self.frame_after, text="Processed Image")
        self.imgLabel_after.pack()

    def open_img(self):
        """開啟圖片"""
        file_types = [('Image Files', '*.png;*.jpg;*.jpeg')]
        path = filedialog.askopenfilename(filetypes=file_types)
        if not path:
            print("No file chosen!")
            return

        self.image = Image.open(path)
        self.display_image(self.image, self.imgLabel_before)
        self.process_img()

    def display_image(self, img, label):
        """顯示圖片"""
        width = 500
        height = int(img.height * width / img.width)
        img = img.resize((width, height))
        img = ImageTk.PhotoImage(img)
        label.configure(image=img)
        label.image = img

    def process_img(self):
        """處理圖片"""
        # 分割圖片
        original_image, mask = self.processor.segment_image(self.image)
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)    # 確保遮罩和原始圖像大小一致。將遮罩從模型的輸出大小（T.Resize）縮放到與原始圖像一致的大小
        largest_mask = self.processor.get_largest_connected_area(mask)

        # 將最大區域遮罩應用到原始圖像
        masked_image = cv2.bitwise_and(original_image, original_image, mask=largest_mask)

        # 將 masked_image 轉換為 RGB 格式以適配 MediaPipe
        rgb_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

        # 檢測人臉特徵點
        results = face_mesh.process(rgb_masked_image)

        # 檢查是否檢測到人臉特徵點
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    # 計算特徵點在圖像中的座標
                    x = int(landmark.x * masked_image.shape[1])
                    y = int(landmark.y * masked_image.shape[0])
                    
                    # 在每個五官的特徵點上繪製圓點
                    cv2.circle(masked_image, (x, y), 1, (0, 255, 0), -1)  # 綠色圓點，大小為 1
        else:
            print("No facial landmarks detected")
         
        # 取得情緒
        emotion = self.processor.emotion_recognize(masked_image)

        # 調整背景
        background = self.processor.adjust_background(original_image, largest_mask, emotion)

        # 提取人像
        people = np.where(largest_mask[..., None] == 255, original_image, 0)

        

        # 顯示結果
        final_image = people + background

        left_eye_center = face_landmarks.landmark[33]  # 左眼中央
        right_eye_center = face_landmarks.landmark[263]  # 右眼中央
        # 計算眼睛位置，這裡將眼睛位置下方畫圓表示眼淚
        left_x = int(left_eye_center.x * final_image.shape[1])
        left_y = int(left_eye_center.y * final_image.shape[0])

        right_x = int(right_eye_center.x * final_image.shape[1])
        right_y = int(right_eye_center.y * final_image.shape[0])

        # 在左眼下方繪製水滴型眼淚
        cv2.ellipse(final_image, 
                (left_x, left_y + 10),  # 眼淚位置 (x, y)
                (5, 10),  # 水滴的長短軸
                0,  # 旋轉角度
                0,  # 起始角度
                180,  # 結束角度，這裡設置為 180 來畫出上半部分
                (255, 0, 0),  # 顏色：藍色
                -1)  # 填充顏色

        # 在右眼下方繪製水滴型眼淚
        cv2.ellipse(final_image, 
            (right_x, right_y + 10),  # 眼淚位置 (x, y)
            (5, 10),  # 水滴的長短軸
            0,  # 旋轉角度
            0,  # 起始角度
            180,  # 結束角度，這裡設置為 180 來畫出上半部分
            (255, 0, 0),  # 顏色：藍色
            -1)  # 填充顏色


        # # 顯示結果
        # cv2.imshow('Masked Image with Facial Landmarks', masked_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        final_image_pil = Image.fromarray(final_image)
        self.display_image(final_image_pil, self.imgLabel_after)


# 主程式
if __name__ == "__main__":
    app = tk.Tk()
    gui = GUI(app)
    app.mainloop()
