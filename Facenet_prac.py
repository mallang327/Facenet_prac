import cv2
import torch
from PIL import Image, ImageDraw
import numpy as np
from facenet_pytorch import MTCNN
import os
from threading import Thread
import matlab.engine
import matlab
import pyaudio
import wave
import time
import matplotlib.pyplot as plt
import scipy.io
import scipy.io.wavfile
import subprocess

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
'''
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
'''
class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

        self.open = True
        self.fps = 6
        self.frameSize = (640, 480)
        self.fourcc = "MJPG"
        self.video_filename = "temp_video.avi"
        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1
        self.start_time = time.time()
        self.cap = cv2.VideoCapture(0)

    def _draw(self, frame, boxes, probs, landmarks):
        # Draw landmarks and boxes for each face detected

        for box, prob, ld in zip(boxes, probs, landmarks):
            if prob > 0.9:

                # zip array by sequence (boxes[0], probs[0], landmarks[0])
                # Draw rectangle on frame
                # boxes[0] has 4 points !([0]~[3]) # 0,0,255 means red color
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                              (0, 0, 255), thickness=1)
                '''
                # Show probability
                cv2.putText(frame, str(prob), (int(box[2]), int(box[3])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2, cv2.LINE_AA)
                '''
                # Detect mouth
                mouth_ld = []
                mouth_ld.append(int((ld[3][0] + ld[4][0]) / 2))
                mouth_ld.append(int((ld[3][1] + ld[4][1]) / 2))
                # Detect Degree
                deg_x = (mouth_ld[0]/len(frame[0]))*180
                deg_x = int(deg_x)
                #deg_y = mouth_ld[1]/len(frame)
                #deg_y = int(deg_y)
                deg_all.append(deg_x)
                # Show Degree
                cv2.putText(frame, 'degree: ' + str(deg_x), (int(box[0]), int(box[3])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                            1, cv2.LINE_AA)
                # Draw landmarks
                cv2.circle(frame, tuple(map(int, tuple(ld[0]))), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(map(int, tuple(ld[1]))), 5, (0, 0, 255), -1)
                #cv2.circle(frame, tuple(map(int, tuple(ld[2]))), 5, (0, 0, 255), -1)
                #cv2.circle(frame, tuple(map(int, tuple(ld[3]))), 5, (0, 0, 255), -1)
                #cv2.circle(frame, tuple(map(int, tuple(ld[4]))), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(map(int, tuple(mouth_ld))), 5, (0, 0, 255), -1)

                #print("ld",ld)
                #print('mouth',mouth_ld)
                # Show Location
                cv2.putText(frame, int(box[0]), (int(box[0]), int(box[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2, cv2.LINE_AA)

        return frame
    '''
    def audio_run(self):
        mat_eng = matlab.engine.start_matlab()
        mat_eng.audio_realtime2()
    '''
    def run(self):
        global deg_all
        deg_all = []

        # Run the FaceDetector and draw landmarks and boxes around detected faces
        #self.cap = cv2.VideoCapture(0) # 0th webcam device
        timer_start = time.time()
        while (self.open == True):

            timer_end = time.time()
            if timer_end - timer_start > 8:
                break

            ret, frame = self.cap.read() # ret: if frame, return True, if not, return False

            try:
                # detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                # /model/MTCNN.py line272 (class mtcnn def detect)
                self.video_out.write(frame)
                #print str(counter) + " " + str(self.frame_counts) + " frames written " + str(timer_current)
                self.frame_counts += 1
                # counter += 1
                #timer_current = time.time() - timer_start
                time.sleep(0.16)
                # draw on frame

                self._draw(frame, boxes, probs, landmarks)
            except:
                pass
            dst = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            # Show the frame
            cv2.imshow('Face Detection', dst)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        global deg_avg
        ndeg_all = np.array(deg_all)
        deg_avg = np.mean(ndeg_all)
        print(deg_avg)

    def start(self):
        fcd = Thread(target=self.run)
        fcd.start()

    def stop(self):
        self.open = False
        self.video_out.release()
        self.cap.release()
        cv2.destroyAllWindows()
        return deg_avg


class AudioRecorder():
    # Audio class based on pyAudio and Wave
    def __init__(self):

        self.open = True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 8
        self.format = pyaudio.paInt16
        self.audio_filename = "temp_audio.wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      input_device_index=1,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []


    # Audio starts being recorded
    def record(self):

        timer_start = time.time()
        while(self.open == True):
            timer_end = time.time()
            if timer_end - timer_start > 8:
                break
            data = self.stream.read(self.frames_per_buffer)
            self.audio_frames.append(data)
            if self.open==False:
                break
        pass

    # Finishes the audio recording therefore the thread too
    def stop(self):

        if self.open==True:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

        pass

    # Launches the audio recording function using a thread
    def start(self):
        audio_thread = Thread(target=self.record)
        audio_thread.start()

# Run the app
mtcnn = MTCNN(keep_all=True, device=device)
# mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
audio_thread = AudioRecorder()


#p2 = Thread(target = fcd.audio_run())
fcd.start()
audio_thread.start()
timer_start = time.time()
while True:
    timer_end = time.time()
    print(timer_end - timer_start)
    time.sleep(0.5)
    if timer_end - timer_start > 10:
        break
audio_thread.stop()
deg_test = fcd.stop()
print("deg_test", deg_test)
'''
frame_counts = p1.frame_counts
elapsed_time = time.time() - p1.start_time
recorded_fps = frame_counts / elapsed_time
print("total frames " + str(frame_counts))
print("elapsed time " + str(elapsed_time))
print("recorded fps " + str(recorded_fps))
'''
myAudioFilename = "temp_audio.wav"
wavedata = 'C:/Users/hr02w/DesignProject/Facedetection/facenet/'+myAudioFilename
sampleRate, audioBuffer = scipy.io.wavfile.read(wavedata)

duration = len(audioBuffer)/sampleRate

time = np.arange(0,duration,1/sampleRate) #time vector

plt.plot(time,audioBuffer)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title(myAudioFilename)
plt.show()


