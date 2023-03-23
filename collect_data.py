import cv2
import os

count = 64
label = input('Enter label: ')
dir_path = os.path.join('train_data',label)

if not os.path.isdir(dir_path):
    os.mkdir(dir_path)
cap = cv2.VideoCapture(0)

leap = 1
cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

while cap.isOpened() and count:
    isSuccess, frame = cap.read()
    if leap%2:
        path = str(dir_path+'/{}_{}.jpg'.format(label,str(50 - count)))
        cv2.imwrite(path,frame)
        count-=1
    leap+=1
    cv2.imshow('Face Capturing', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()