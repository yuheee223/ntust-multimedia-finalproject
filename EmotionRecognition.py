# https://github.com/jhan15/facial_emotion_recognition
import tensorflow as tf

import numpy as np
import cv2
import mediapipe as mp
import time
import glob

from keras.api.models import Sequential
from keras.api.layers import Rescaling
# from keras.layers.experimental.preprocessing import Rescaling
from keras.api.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.api.layers import BatchNormalization
from keras.api.losses import categorical_crossentropy
from keras.api.optimizers import Adam

# Parameters & Model
emotions = {
    0: ['Angry', (0,0,255), (255,255,255)],
    1: ['Disgust', (0,102,0), (255,255,255)],
    2: ['Fear', (255,255,153), (0,51,51)],
    3: ['Happy', (153,0,153), (255,255,255)],
    4: ['Sad', (255,0,0), (255,255,255)],
    5: ['Surprise', (0,255,0), (255,255,255)],
    6: ['Neutral', (160,160,160), (255,255,255)]
}
num_classes = len(emotions)
input_shape = (48, 48, 1)
weights_1 = 'model/vggnet.h5'
weights_2 = 'model/vggnet_up.h5'

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

model_1 = VGGNet(input_shape, num_classes, weights_1)
model_1.load_weights(model_1.checkpoint_path)

model_2 = VGGNet(input_shape, num_classes, weights_2)
model_2.load_weights(model_2.checkpoint_path)


# Inference
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def detection_preprocessing(image, h_max=360):
    h, w, _ = image.shape
    if h > h_max:
        ratio = h_max / h
        w_ = int(w * ratio)
        image = cv2.resize(image, (w_,h_max))
    return image

def resize_face(face):
    x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
    return tf.image.resize(x, (48,48))

def recognition_preprocessing(faces):
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x

def inference(image):
    H, W, _ = image.shape
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    if results.detections:
        faces = []
        pos = []
        for detection in results.detections:
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
    
        x = recognition_preprocessing(faces)

        y_1 = model_1.predict(x)
        y_2 = model_2.predict(x)
        l = np.argmax(y_1+y_2, axis=1)

        for i in range(len(faces)):
            cv2.rectangle(image, (pos[i][0],pos[i][1]),
                            (pos[i][2],pos[i][3]), emotions[l[i]][1], 2, lineType=cv2.LINE_AA)
            
            cv2.rectangle(image, (pos[i][0],pos[i][1]-20),
                            (pos[i][2]+20,pos[i][1]), emotions[l[i]][1], -1, lineType=cv2.LINE_AA)
            
            cv2.putText(image, f'{emotions[l[i]][0]}', (pos[i][0],pos[i][1]-5),
                            0, 0.6, emotions[l[i]][2], 2, lineType=cv2.LINE_AA)
    
    return image


# # ----------Image----------
# def infer_single_image(path):
#     image = cv2.imread(path)
#     image = detection_preprocessing(image)
#     result = inference(image)
#     cv2.imwrite('img/out/out.jpg', result)

# def infer_multi_images(paths):
#     for i, path in enumerate(paths):
#         image = cv2.imread(path)
#         image = detection_preprocessing(image)
#         result = inference(image)
#         cv2.imwrite('img/out/out_'+str(i)+'.jpg', result)

# infer_single_image('img/multi_1.jpg')
# out = cv2.imread('img/out/out.jpg')
# cv2.imshow(out)

# paths = np.sort(np.array(glob.glob('img/*.jpg')))
# infer_multi_images(paths)
# out_paths = np.sort(np.array(glob.glob('img/out/*.jpg')))
# for path in out_paths:
#     image = cv2.imread(path)
#     cv2.imshow(image)


# # ----------Video----------
# video = 'video/nj_bubblegum.mp4'
# cap = cv2.VideoCapture(video)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# target_h = 360
# target_w = int(target_h * frame_width / frame_height)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 輸出編碼格式
# out = cv2.VideoWriter('video/out.mp4', fourcc, fps, (target_w, target_h))
# # out = cv2.VideoWriter('video/out.mp4',cv2.VideoWriter_fourcc('M','J','P','G'),
#                     #   fps, (target_w,target_h))

# while True:
#     success, image = cap.read()
#     if success:
#         image = detection_preprocessing(image)
#         result = inference(image)
#         out.write(result)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
    
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# # ----------Cam----------
# # 執行人臉偵測
# cap = cv2.VideoCapture(0)               # 讀取攝影鏡頭
# while True:
#     success, image = cap.read()
#     if not success:
#         print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
#         break
#     image = detection_preprocessing(image)
#     result = inference(image)
#     cv2.imshow('oxxostudio', result)     # 如果讀取成功，顯示該幀的畫面
#     if cv2.waitKey(10) == ord('q'):     # 每一毫秒更新一次，直到按下 q 結束
#         break
# cap.release()           # 釋放資源
cv2.destroyAllWindows() # 結束所有視窗