import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image

"""
# 定義該情緒的中文字
text_obj={
    'angry': '生氣',
    'disgust': '噁心',
    'fear': '害怕',
    'happy': '開心',
    'sad': '難過',
    'surprise': '驚訝',
    'neutral': '正常'
}
"""

img_file = 'img/haerin.jpg'

image = Image.open(img_file).convert("RGB")
img = np.array(image)    # 原始圖像作為 NumPy 陣列

# img = cv2.imread(img_file) # 讀取圖片
try:
    emotion = DeepFace.analyze(img, actions=['emotion']) # 辨識圖片人臉資訊，取出情緒資訊
    print(emotion[0]['dominant_emotion'])
except:
    pass

# cv2.imshow('oxxostudio', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()