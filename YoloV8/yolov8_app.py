import cv2
import imutils

import os
if "vg_4" not in os.listdir("./RealTimeMOT/YoloV8/Train5") :
  os.mkdir("./RealTimeMOT/YoloV8/Train5/vg_4")
  os.mkdir("./RealTimeMOT/YoloV8/Train5/vg_4/det")
  os.mkdir("./RealTimeMOT/YoloV8/Train5/vg_4/img1")

if "output" not in os.listdir("./RealTimeMOT/YoloV8/Train5") :
  os.mkdir("./RealTimeMOT/YoloV8/Train5/output")

det = open("./RealTimeMOT/YoloV8/Train5/vg_4/det/det.txt","w")
det.close()

from ultralytics import YOLO
model = YOLO("./RealTimeMOT/YoloV8/Train5/runs/detect/train/weights/best.pt") 

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


LABELS = {0: u'commercial vessel', 1: u'recreational vessel', 2: u'sailboat', 3: u'Container Ship', 4: u'Cruise', 5: u'DDG', 6: u'Recreational', 7: u'Sailboat', 8: u'Submarine', 9: u'Tug'}
COLORS = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(200,50,20),(150,150,0),(0,30,180),(190,40,200),(250,0,0),(0,80,40),(200,240,205)]
CONF_TRESHOLD = 0.10


## Création de seqinfo.ini
vs = cv2.VideoCapture("./inputVideos/vg/vg_4.mp4")

if not vs.isOpened() :
  raise SystemError("Couldn't read the input video")

seq = open("./RealTimeMOT/YoloV8/Train5/vg_4/seqinfo.ini","w")
seq.write("[Sequence]\nname=vg_4\nimDir=img1")
seq.close()

# FrameRate
try:
	prop = cv2.cv.CV_CAP_PROP_FPS if imutils.is_cv2() \
		else cv2.CAP_PROP_FPS
	fps = float(vs.get(prop))
	seq = open("./RealTimeMOT/YoloV8/Train5/vg_4/seqinfo.ini","a")
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
	seq = open("./RealTimeMOT/YoloV8/Train5/vg_4/seqinfo.ini","a")
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
	seq = open("./RealTimeMOT/YoloV8/Train5/vg_4/seqinfo.ini","a")
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
	seq = open("./RealTimeMOT/YoloV8/Train5/vg_4/seqinfo.ini","a")
	seq.write("\nimHeight={}".format(height))
	seq.close()
	print("[INFO] frames of {} height".format(height))

except:
	print("An error occurred while trying to determine the height")
	total = -1

seq = open("./RealTimeMOT/YoloV8/Train5/vg_4/seqinfo.ini","a")
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

acc = 0
while acc < total :
  (grabbed, frame) = vs.read()

  if grabbed:
    acc +=1
    # save blank frame in images folder
    cv2.imwrite("./RealTimeMOT/YoloV8/Train5/vg_4/img1/"+zeros[:i-len(str(acc))]+str(acc)+".jpg", frame)

    # predict bounding boxes on frame
    results = model.predict(frame, verbose=False)

    # draw boudning boxes on frame
    boxes = results[0].boxes.data
    for box in boxes:
      label = LABELS[int(box[-1])+1]
      if box[-2] > CONF_TRESHOLD :
          color = COLORS[int(box[-1])]
          box_label(frame, box, label, color)

          # save detection in det.txt
          det = open("./RealTimeMOT/YoloV8/Train5/vg_4/det/det.txt","a")
          bbMOT = formatMOT(acc,box)
          print(bbMOT)
          det.write(bbMOT)
          det.write("\n")
          det.close()
          
    # save frame in video writer
    if writer is None:
      fourcc = cv2.VideoWriter_fourcc(*"MJPG")
      writer = cv2.VideoWriter('./RealTimeMOT/YoloV8/Train5/output/detections_vg_4.avi', fourcc=fourcc, fps=fps,
      frameSize=(frame.shape[1], frame.shape[0]), isColor=True)
    

  writer.write(frame)
writer.release()