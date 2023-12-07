import cv2
import imutils
import os

## Import of last trained detection model and definition of its parameters
from ultralytics import YOLO
model = YOLO("D:/LEVRAUDLaura/Dev/LowerPythonEnv/RealTimeMOT/YoloV8/Train15/runs/detect/train/weights/best.pt")

LABELS = {0: u'__background__', 1: u'sailboat'}
COLORS = [(89, 161, 197),(67, 161, 255)]
CONF_TRESHOLD = 0.10

## Selection of the images that will be treated
path = "D:/LEVRAUDLaura/Dev/LowerPythonEnv/inputVideos/bateau/bateau_1/"
images = [path+image_file for image_file in os.listdir(path)]

## Creation of results folder and results files


## Creation of sequence informations file seqinfo.ini


## Format detection results
def formatMOT(frame,box) :
  bbMOT = "{}".format(int(frame))
  bbMOT = bbMOT + ",-1,"

  x = round(float(box[0]),2)
  bbMOT = bbMOT + "{},".format(float(x))

  y = round(float(box[1]),2)
  bbMOT = bbMOT + "{},".format(float(y))

  width = round(abs(float(box[2]-box[0])),2)
  bbMOT = bbMOT + "{},".format(float(width))

  height = round(abs(float(box[3]-box[1])),2)
  bbMOT = bbMOT + "{},".format(float(height))

  score = round(100 * float(box[-2]),1)
  bbMOT = bbMOT + "{},-1,-1".format(float(score))
  
  return bbMOT

### Tracking algorithm
acc = 1
for frame in images :
    ## Detection
    predictions = model.predict(frame, verbose=False)
    boxes = predictions[0].boxes.data
    
    ## Format detections of interest
    detections = []
    for box in boxes:
      if box[-2] > CONF_TRESHOLD :
        detections.append(formatMOT(acc,box))
    
    
    acc += 1