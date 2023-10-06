print("test")


import cv2
print(cv2.__version__)


video = cv2.VideoCapture("inputVideos/bateau_1.mp4")
print(video.isOpened())


ok, frame = video.read()
print(ok)


bbox = cv2.selectROI(frame,False)


tracker = cv2.legacy.TrackerKCF_create()