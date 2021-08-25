import cv2
import torch
from PIL import Image, ImageDraw
import numpy as np
from facenet_pytorch import MTCNN
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):
        # Draw landmarks and boxes for each face detected

        for box, prob, ld in zip(boxes, probs, landmarks):
            # zip은 배열 순서대로 하나씩 묶어줌 (boxes[0], probs[0], landmarks[0])
            # Draw rectangle on frame
            # boxes[0]에 4개 point 존재!([0]~[3]) # 0,0,255는 red 색깔

            cv2.rectangle(frame, (int(box[0]),int(box[1])),(int(box[2]),int(box[3])),
                          (0,0,255),thickness=2)

            # Show probability
            cv2.putText(frame, str(prob),(int(box[2]),int(box[3])),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),
                       2, cv2.LINE_AA)
            # Draw landmarks
            cv2.circle(frame, tuple(map(int,tuple(ld[0]))), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(map(int,tuple(ld[1]))), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(map(int,tuple(ld[2]))), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(map(int,tuple(ld[3]))), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(map(int,tuple(ld[4]))), 5, (0, 0, 255), -1)


            # Show Location
            cv2.putText(frame, int(box[0]),(int(box[0]),int(box[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),
                       2, cv2.LINE_AA)            



        return frame

    def run(self):
        # Run the FaceDetector and draw landmarks and boxes around detected faces
        img2 = 'C:/Users/hr02w/DesignProject/Facedetection/facenet/facenet_pytorch/data/test_images/shea_whigham/1.jpg'
        img = cv2.imread(img2, cv2.IMREAD_COLOR)

        # detect face box, probability and landmarks
        boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
        # /model/MTCNN.py line272 (class mtcnn def detect)

        # draw on frame
        self._draw(img, boxes, probs, landmarks)


        # Show the frame
        cv2.imshow('Face Detection', img)

        cv2.waitKey(0)

        cv2.destroyAllWindows()


# Run the app
mtcnn = MTCNN(keep_all=True, device=device)
# mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
fcd.run()
            
'''
img = 'C:/Users/hr02w/DesignProject/Facedetection/test.jpg'
src = cv2.imread(img, cv2.IMREAD_COLOR)
src = src.astype(np.uint8)
'''