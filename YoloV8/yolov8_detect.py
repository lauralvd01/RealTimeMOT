
import os
import cv2
import imutils
import datetime

from ultralytics import YOLO
import random

class GetInfos :
  def __init__(self,video_path,output_dir) -> None:
    self.video_path = video_path
    self.video_file = video_path.strip().split(os.path.sep)[-1]
    self.video_name = self.video_file.split('.')[-2]
    self.video_ext = self.video_file.split('.')[-1]
    self.video = None
    self.output_dir = output_dir
  
  def openVideo(self) :
    self.video = cv2.VideoCapture(self.video_path)
    if not self.video.isOpened() :
      raise SystemError("Couldn't read the input video")
  
  def getVideo(self) :
    if self.video is None :
      self.openVideo()
    return self.video
  
  def getFrameRate(self) :
    try:
      if imutils.is_cv2() :
        prop = cv2.cv.CV_CAP_PROP_FPS
      else :
        prop = cv2.CAP_PROP_FPS
      self.fps = float(self.getVideo().get(prop))
      print(f"[INFO] {format(self.fps)} frames per second in video")
      
    except:
      raise Exception("An error occurred while trying to determine the fps")
  
  def getTotalFrames(self) :
    try:
      if imutils.is_cv2() :
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
      else :
        prop = cv2.CAP_PROP_FRAME_COUNT
      self.total_frames = int(self.getVideo().get(prop))
      print(f"[INFO] {format(self.total_frames)} total frames in video")
      
    except:
      raise Exception("An error occurred while trying to determine the total number of frames")
  
  def getFrameWidth(self) :
    try:
      if imutils.is_cv2() :
        prop = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
      else :
        prop = cv2.CAP_PROP_FRAME_WIDTH
      self.frame_width = int(self.getVideo().get(prop))
      print(f"[INFO] frames of {format(self.frame_width)} width")
      
    except:
      raise Exception("An error occurred while trying to determine the frame width")
  
  def getFrameHeight(self) :
    try:
      if imutils.is_cv2() :
        prop = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
      else :
        prop = cv2.CAP_PROP_FRAME_HEIGHT
      self.frame_height = int(self.getVideo().get(prop))
      print(f"[INFO] frames of {format(self.frame_height)} height")
      
    except:
      raise Exception("An error occurred while trying to determine the frame height")
  
  def saveInfos(self) :
    self.openVideo()
    self.getFrameRate()
    self.getTotalFrames()
    self.getFrameWidth()
    self.getFrameHeight()
    infos = f"""[{datetime.date.today()}]
    name={self.video_name}
    imDir=img1
    frameRate={format(self.fps)}
    seqLength={format(self.total_frames)}
    imWidth={format(self.frame_width)}
    imHeight={format(self.frame_height)}
    imExt=.jpg
    """
    file = open(self.output_dir+os.path.sep+"seqinfo.ini","w")
    file.write(infos)
    file.close()

class YoloModel :
  def __init__(self,model=None) -> None:
    if model is None :
      model = YOLO('yolov8n.pt')
    self.model_pt = YOLO(model)
    self.model_onnx = None
    self.labels = self.model_pt.names
    self.colors = [(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)) for i in range(len(self.labels))]
  
  def getOnnx(self) :
    if self.model_onnx is None :
      self.model_onnx = self.model_pt.export(format='onnx')
    return self.model_onnx

class Detect :
  
  def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)) :
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
  
  def __init__(self,video_path,output_dir,model,conf) -> None:
    self.video = GetInfos(video_path,output_dir)
    self.model = YoloModel(model)
    self.conf = conf
    self.video_output = None
    self.zeros = None
    det = open(output_dir+os.path.sep+"det.txt","w")
    det.close()
    if "img1" not in os.listdir(output_dir) :
      os.mkdir(output_dir+os.path.sep+"img1")
  
  def openVideoOutput(self) :
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    self.video_output = cv2.VideoWriter(self.video.output_dir+os.path.sep+'video_output.avi', fourcc=fourcc, fps=self.video.fps, frameSize=(self.video.frame_width, self.video.frame_height), isColor=True)
    if not self.video_output.isOpened() :
      raise SystemError("Couldn't open the output video")
  
  def getVideoOutputWriter(self) :
    if self.video_output is None :
      self.openVideoOutput()
    return self.video_output
  
  def formatFrameNumber(self,number) :
    if self.zeros is None :
      i = 0
      tot = self.video.total_frames
      while tot > 0 :
        tot = tot//10
        i += 1
      zeros = "0"*i
      self.zeros = zeros
    return self.zeros[:len(self.zeros)-len(str(number))]+str(number)
  
  def run(self) :
    self.video.saveInfos()
    writer = self.getVideoOutputWriter()
    
    frame_number = 0
    while frame_number < self.video.total_frames :
      frame_number += 1
      number = self.formatFrameNumber(frame_number)
      
      grabbed, frame = self.video.getVideo().read()
      if not grabbed :
        raise Exception("Couldn't read frame number "+number)
      
      print("Frame "+number+f" over {format(self.video.total_frames)}")
      cv2.imwrite(self.video.output_dir+os.path.sep+"img1"+os.path.sep+number+".jpg",frame)
      
      results = self.model.model_pt.predict(frame, verbose=False)
      
      boxes = results[0].boxes.data
      for box in boxes:
        label = self.model.labels[int(box[-1])]
        if box[-2] > CONF_TRESHOLD :
            color = self.model.colors[int(box[-1])]
            Detect.box_label(frame, box, label, color)

            det = open(self.video.output_dir+os.path.sep+"det.txt","a")
            bbMOT = Detect.formatMOT(number,box)
            det.write(bbMOT)
            det.write("\n")
            det.close()
      
      writer.write(frame)
    
    writer.release()
  

VIDEO_PATH = "D:/LEVRAUDLaura/Data/H3_Depart_court.mov"
OUTPUT_DIR = "D:/LEVRAUDLaura/Dev/LowerPythonEnv/RealTimeMOT/YoloV8/train_cd_best_and_new_model/output"
MODEL = "D:/LEVRAUDLaura/Dev/LowerPythonEnv/RealTimeMOT/YoloV8/train_cd_best_and_new_model/weights/best.pt"
CONF_TRESHOLD = 0.10

detect = Detect(VIDEO_PATH,OUTPUT_DIR,MODEL,CONF_TRESHOLD)
detect.run()