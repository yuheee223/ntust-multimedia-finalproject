import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import models
from PIL import Image
from torchvision import transforms as T
import numpy as np
import cv2
from deepface import DeepFace

# 反正規化
def denormalize(tensor, mean, std):
    # 建一個新的transform來反正規化
    denormalize_transform = T.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    return denormalize_transform(tensor)

# 根據情緒選擇背景顏色
def get_background_color(emotion):
    emotion_to_color = {
        'happy': (255, 223, 186),   # 橘色
        'sad': (135, 206, 235),     # 藍色
        'angry': (255, 105, 97),    # 紅色
        'neutral': (211, 211, 211), # 灰色
    }
    return emotion_to_color.get(emotion, (255, 255, 255))  # 默認返回白色

# 更改背景顏色
def adjust_hue(image, target_color):
    # 將圖片從 BGR 轉換為 HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 獲取目標顏色的 H 值 (色相)
    target_hue = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)[0][0][0]
    
    # 計算當前圖片的色相
    current_hue = np.mean(hsv_image[:, :, 0])
    
    # 計算色相偏移
    hue_shift = target_hue - current_hue
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180  # 保持色相範圍在 [0, 180]

    # 轉換回 BGR
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return result_image

img_file = "img/wooyoung.png"

if __name__ == '__main__':
    # 載入預訓練的DeepLabv3模型
    model = deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, num_classes=21)
    model.eval()

    # 模型正規化參數
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 將圖片轉換為模型需要的輸入格式
    transform = T.Compose([
        T.Resize(512),  # 需要固定輸入大小
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # 載入圖片
    image = Image.open(img_file).convert("RGB")
    original_image = np.array(image)    # 原始圖像作為 NumPy 陣列
    image_tensor = transform(image).unsqueeze(0)    # 模型輸入

    # 預測輸出
    with torch.no_grad():
        output = model(image_tensor)['out']
        output_predictions = F.softmax(output, dim=1).argmax(dim=1)

    # 提取分割遮罩
    output_predictions = output_predictions.squeeze(0).cpu().numpy()
    mask = np.uint8(output_predictions == 15) * 255  # 提取人像 (類別 15 為人像)

    # 確保遮罩和原始圖像大小一致
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)    # 將遮罩從模型的輸出大小（T.Resize）縮放到與原始圖像一致的大小

    # 找到最大連通區域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_resized, connectivity=8)
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 忽略背景 (index 0)
    largest_mask = np.uint8(labels == max_label) * 255  # 只保留最大區域

    # 將最大區域遮罩應用到原始圖像
    masked_image = cv2.bitwise_and(original_image, original_image, mask=largest_mask)

    # 保存結果
    masked_image_pil = Image.fromarray(masked_image)
    # masked_image_pil.save('masked_output.png')

    # 情緒識別
    try:
        emotion = DeepFace.analyze(masked_image, actions=['emotion']) # 辨識圖片人臉資訊，取出情緒資訊
        dominant_emotion = emotion[0]['dominant_emotion']
        print(emotion[0]['dominant_emotion'])

        # 根據情緒選擇背景顏色
        background_color = get_background_color(dominant_emotion)
    except:
        print("can't get emotion")

    # 顯示結果
    # masked_image_pil.show()

    # 根據情緒選擇目標背景顏色
    background_color = get_background_color(dominant_emotion)

    # 提取背景
    background_mask = np.uint8(largest_mask == 0) * 255  # 反轉遮罩，背景區域為255，人像區域為0
    background = cv2.bitwise_and(original_image, original_image, mask=background_mask)

    # 使用示範圖像的背景區域進行色調調整
    background = adjust_hue(background, background_color)
    # Image.fromarray(background).show()

    # 提取人像區域
    people = np.where(largest_mask[..., None] == 255, original_image, 0)  # 保留人像，背景設為0

    final_image = people + background

    # # 顯示結果
    final_image_pil = Image.fromarray(final_image)
    final_image_pil.show()

