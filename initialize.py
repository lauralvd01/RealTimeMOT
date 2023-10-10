import cv2
import sys
import os
import pandas

def verifBbox(bbox,min_x=0,max_x=1,min_y=0,max_y=1,min_width=0,max_width=1,min_height=0,max_height=1):
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[0] + bbox[2])
    y2 = int(bbox[1] + bbox[3])
    width = x2-x1
    height = y2-y1
    if min_x < x1 and x1 < max_x:
        if min_x < x2 and x2 < max_x:
            if min_y < y1 and y1 < max_y:
                if min_y < y2 and y2 < max_y:
                    if min_width < width and width < max_width:
                        if min_height < height and height < max_height:
                            return True
    return False


if __name__ == '__main__' :

    # Select the type of the videos
    videos_folder = "inputVideos/"
    videos_type = "velo"
    videos = [name for name in os.listdir('./'+videos_folder+videos_type)]
    
    # Iterate on the video files present in the folder
    for video_file in videos:
        parts = video_file.split('.')
        video_file_name = parts[0]
        video_file_ext = parts[1]
        
        # Read video
        video = cv2.VideoCapture(videos_folder+videos_type+'/'+video_file_name+'.'+video_file_ext)
    
        # Exit if video is not opened
        if not video.isOpened():
            print("Could not open video")
            sys.exit()
        else:
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
        # Read first frame
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        
        # Select all targets
        print("Select all targets, pressing Enter for each, then quit with one more Enter")
        targets = []
        while True:
            bbox = cv2.selectROI(frame,False)
            
            # Exit if no bbox
            if not verifBbox(bbox,max_x=width,max_y=height,max_width=width,max_height=height):
                break
            
            targets.append(str(bbox))
        
        # Register targets in an Excel file
        data_folder = "dataFiles/init/" + videos_type + "/"
        data = pandas.DataFrame({"Bbox":targets})
        data.to_excel(data_folder+video_file_name+'.xlsx',sheet_name = 'targets', index= [i+1 for i in range(len(targets))])
        