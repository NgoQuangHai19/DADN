from tensorflow.keras import models
import numpy as np
import os 
from PIL import Image
import cv2
import keyboard
import sys

data_path = 'train_data'
#load_model
face_id_model = models.load_model('face_id.h5')
original_labels = os.listdir(data_path)
cam = cv2.VideoCapture(0) 

while True:
    isSuccess, frame = cam.read()
    if not isSuccess:
        print("fail to grab frame, try again")
        break
    img = Image.fromarray(frame)
    if img:
        face_image = img.resize((128, 128))
        face_numpy = np.array(face_image, 'uint8')
        face_numpy = np.expand_dims(face_numpy, axis=0)
        result = face_id_model.predict(face_numpy/255)
        frame = cv2.putText(frame,'Recognize: {}'.format(original_labels[np.argmax(result[0])]), (12,45), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,0), 2, cv2.LINE_8)
    cv2.imshow("Face Recognition", frame)
    
    try:
        if keyboard.is_pressed('esc'): 
            print('Esc pressed, closing...')
            break 
    except:
        continue
cam.release()
cv2.destroyAllWindows()

sys.exit(0)