import cv2
import torch
from PIL import Image, ImageDraw
import numpy as np
from facenet_pytorch import MTCNN
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Run the FaceDetector and draw landmarks and boxes around detected faces
img2 = 'C:/Users/hr02w/DesignProject/Facedetection/facenet/facenet_pytorch/data/test_images/shea_whigham/1.jpg'
img = cv2.imread(img2, cv2.IMREAD_COLOR)

# detect face box, probability and landmarks
boxes, probs, landmarks = MTCNN.detect(img, landmarks=True)
# /model/MTCNN.py line272 (class mtcnn def detect)
print(boxes)
print(probs)
print(landmarks)

#print(landmarks[0])
#print('/n')
#print(landmarks[0][0])
ld = []
ld.append(955)
ld.append(589)
print('/n')
print(tuple(ld))
cv2.rectangle(img, (846,390),(1260,906),(0,0,255),thickness=2)
cv2.circle(img, tuple(ld), 5, (0, 0, 255), -1)
'''
frame = img
for box, prob, ld in zip(boxes, probs, landmarks):
    # zip은 배열 순서대로 하나씩 묶어줌 (boxes[0], probs[0], landmarks[0])
    # Draw rectangle on frame
    # boxes[0]에 4개 point 존재!([0]~[3]) # 0,0,255는 red 색깔

    cv2.rectangle(frame, (box[0],box[1]),(box[2],box[3]),
                  (0,0,255),thickness=2)

    # Show probability
    cv2.putText(frame, str(prob),(box[2],box[3]),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),
               2, cv2.LINE_AA)
    # Draw landmarks
    cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
    cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
    cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
    cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
    cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)

    

    
    # Show Location
    cv2.putText(frame, box[0],(box[0],box[1]),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),
               2, cv2.LINE_AA)            
    '''
img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
# Show the frame
cv2.imshow('Face Detection', img)
cv2.waitKey(0)

cv2.destroyAllWindows()

