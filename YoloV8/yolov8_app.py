import cv2
import imutils

import time


import os
if "F1_paysBas" not in os.listdir("./RealTimeMOT/YoloV8/TrainA") :
  os.mkdir("./RealTimeMOT/YoloV8/TrainA/F1_paysBas")
  os.mkdir("./RealTimeMOT/YoloV8/TrainA/F1_paysBas/det")
  os.mkdir("./RealTimeMOT/YoloV8/TrainA/F1_paysBas/img1")

if "output" not in os.listdir("./RealTimeMOT/YoloV8/TrainA") :
  os.mkdir("./RealTimeMOT/YoloV8/TrainA/output")

det = open("./RealTimeMOT/YoloV8/TrainA/F1_paysBas/det/det.txt","w")
det.close()

from ultralytics import YOLO
#model = YOLO("./RealTimeMOT/YoloV8/TrainA/runs/detect/train/weights/best.pt") 
model = YOLO("./RealTimeMOT/YoloV8/Train15/runs/detect/train/weights/best.pt")
model.export(format='onnx')        #export for C# YoloV8 https://github.com/dme-compunet/YOLOv8


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

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


COCO_LABELS = {0: u'__background__', 1: u'person', 2: u'bicycle',3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog', 54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'}
LABELS = {0: u'__background__', 1: u'sailboat'}
F1_LABELS = {0: u'__background__', 1: u'LOVE1', 2: u'VELO1', 3: u'VELO2', 4: u'VELO3', 5: u'VELO4', 6: u'VUSE1', 7: u'VUSE2', 8: u'VUSE3', 9: u'CAR'}
COLORS = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),(155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),(128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),(164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),(181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),(100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),(205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),(119, 24, 48),(73, 8, 110)]
CONF_TRESHOLD = 0.10


## Création de seqinfo.ini
#vs = cv2.VideoCapture("./inputVideos/vg/F1_paysBas.mp4")
#vs = cv2.VideoCapture("D:/LEVRAUDLaura/Data/Do_not_delete_2023_08_52_Super_Series_Barcelona_/230826_52_Super_Series_Barcelona/52 Super Series Barcelona 2023 Race 8.mov")
#vs = cv2.VideoCapture("D:/LEVRAUDLaura/Data/F1-PAYS-BAS-A-FLOUTER.mxf")
vs = cv2.VideoCapture("D:/LEVRAUDLaura/Data/H3_Depart_court.mov")

if not vs.isOpened() :
  raise SystemError("Couldn't read the input video")

seq = open("./RealTimeMOT/YoloV8/TrainA/F1_paysBas/seqinfo.ini","w")
seq.write("[Sequence]\nname=F1_paysBas\nimDir=img1")
seq.close()

# FrameRate
try:
	prop = cv2.cv.CV_CAP_PROP_FPS if imutils.is_cv2() \
		else cv2.CAP_PROP_FPS
	fps = float(vs.get(prop))
	seq = open("./RealTimeMOT/YoloV8/TrainA/F1_paysBas/seqinfo.ini","a")
	seq.write("\nframeRate={}".format(fps))
	seq.close()
	print("[INFO] {} frames per second in video".format(fps))

except:
	print("An error occurred while trying to determine the fps")
	total = -1

# Nombre de frames
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	seq = open("./RealTimeMOT/YoloV8/TrainA/F1_paysBas/seqinfo.ini","a")
	seq.write("\nseqLength={}".format(total))
	seq.close()
	print("[INFO] {} total frames in video".format(total))

except:
	print("An error occurred while trying to determine the total")
	total = -1

# Largeur d'une frame
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_WIDTH if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_WIDTH
	width = int(vs.get(prop))
	seq = open("./RealTimeMOT/YoloV8/TrainA/F1_paysBas/seqinfo.ini","a")
	seq.write("\nimWidth={}".format(width))
	seq.close()
	print("[INFO] frames of {} width".format(width))

except:
	print("An error occurred while trying to determine the width")
	total = -1

# Hauteur d'une frame
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_HEIGHT
	height = int(vs.get(prop))
	seq = open("./RealTimeMOT/YoloV8/TrainA/F1_paysBas/seqinfo.ini","a")
	seq.write("\nimHeight={}".format(height))
	seq.close()
	print("[INFO] frames of {} height".format(height))

except:
	print("An error occurred while trying to determine the height")
	total = -1

seq = open("./RealTimeMOT/YoloV8/TrainA/F1_paysBas/seqinfo.ini","a")
seq.write("\nimExt=.jpg")
seq.close()























# Détections
writer = None
i = 0
tot = total
while tot > 0 :
  tot = tot//10
  i += 1
zeros = "0"*i

start = time.time()
time_moy = 0

acc = 0
while acc < total :
  (grabbed, frame) = vs.read()

  if grabbed:
    acc +=1
    #print(f"Frame {acc} / {total}")
    # save blank frame in images folder
    cv2.imwrite("./RealTimeMOT/YoloV8/TrainA/F1_paysBas/img1/"+zeros[:i-len(str(acc))]+str(acc)+".jpg", frame)

    # predict bounding boxes on frame
    start_pred = time.time()
    results = model.predict(frame, verbose=False)
    end_pred = time.time()
    print(f"Time to process last prediction : {end_pred-start_pred:.6}")
    if acc > 1 : time_moy += end_pred-start_pred

    #draw boudning boxes on frame
    boxes = results[0].boxes.data
    for box in boxes:
      label = F1_LABELS[int(box[-1])+1]
      if box[-2] > CONF_TRESHOLD :
          color = COLORS[int(box[-1])]
          box_label(frame, box, label, color)

          # save detection in det.txt
          det = open("./RealTimeMOT/YoloV8/TrainA/F1_paysBas/det/det.txt","a")
          bbMOT = formatMOT(acc,box)
          print(bbMOT)
          det.write(bbMOT)
          det.write("\n")
          det.close()
          
    #save frame in video writer
    if writer is None:
      fourcc = cv2.VideoWriter_fourcc(*"MJPG")
      #writer = cv2.VideoWriter('./RealTimeMOT/YoloV8/TrainA/output/detections_F1_paysBas.avi', fourcc=fourcc, fps=fps,
      #frameSize=(frame.shape[1], frame.shape[0]), isColor=True)
      
      writer = cv2.VideoWriter('./RealTimeMOT/YoloV8/train_cd_best_model/output/output.avi', fourcc=fourcc, fps=fps,
      frameSize=(frame.shape[1], frame.shape[0]), isColor=True)
      
      
  
  writer.write(frame)
writer.release()
  
end = time.time()
print(f"Time to process whole video : {end-start:.7}")
print(f"Time average : {time_moy/(acc-1):.6}")
