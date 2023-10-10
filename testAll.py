import cv2
import sys
import os
import pandas
from math import ceil

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def createTracker(tracker_type) :
    if int(minor_ver) < 3:
        return cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            return cv2.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            return cv2.legacy.TrackerMIL_create()
        if tracker_type == 'KCF':
            return cv2.legacy.TrackerKCF_create()
        if tracker_type == 'TLD':
            return cv2.legacy.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            return cv2.legacy.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            return cv2.legacy.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        if tracker_type == 'CSRT':
            return cv2.legacy.TrackerCSRT_create()

def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

def drawText(frame, txt, location, color=(110, 170, 50)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 3
    cv2.putText(frame, txt, location, font, fontScale, color, thickness)

def verifBbox(bbox,min_x,max_x,min_y,max_y,min_width,max_width,min_height,max_height):
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

    tracker_types = ['MIL', 'KCF', 'MOSSE', 'CSRT', 'BOOSTING', 'TLD', 'MEDIANFLOW', 'GOTURN']
    video_types = ['bateau']
    
    num_test = 1
    for tracker_type in tracker_types :
        for video_type in video_types :
            
            input_folder = 'inputVideos/' + video_type + '/'
            video_input_files = [name for name in os.listdir('./'+input_folder)]
            print(video_input_files)
            
            for video_input_file in video_input_files :
                
                # Choose video
                parts = video_input_file.split('.')
                video_input_file_name = parts[0]
                video_input_file_extension = '.' + parts[1]
                
                # Read data file
                data_folder = 'dataFiles/init/' + video_type + '/'
                targets_data = pandas.read_excel('./'+data_folder+video_input_file_name+'.xlsx')
                N = len(targets_data)
                
                # Create results data file
                data = {"Test":[num_test+i for i in range(N)],"Type de video":[video_type for i in range(N)], "Point teste":[None for i in range(N)], "Tracker":[tracker_type for i in range(N)], "Input":[video_input_file_name+video_input_file_extension for i in range(N)], "Output":["-1" for i in range(N)], "FPS":[-1.00 for i in range(N)], "Total frames":[-1 for i in range(N)], "Initialisation":["-1" for i in range(N)], "Initialisation scale":["-1" for i in range(N)], "Frames Computed per Second":[-1], "Objet cache":[None for i in range(N)], "Frame before":[None for i in range(N)], "Tracking Failure":[None for i in range(N)], "Frames before":[None for i in range(N)]}
                
                for num_cible in range(len(targets_data)) :
                    
                    # Read video
                    print(input_folder+video_input_file_name+video_input_file_extension)
                    video = cv2.VideoCapture(input_folder+video_input_file_name+video_input_file_extension)

                    # Exit if video is not opened
                    if not video.isOpened():
                        print("Could not open video")
                        sys.exit()
                    else:
                        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = video.get(cv2.CAP_PROP_FPS)
                        total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Read bbox_init
                    bbox_str = targets_data.iat[num_cible,1].split(',')
                    x = bbox_str[0]
                    x = int(x[1:len(x)])
                    y = int(bbox_str[1])
                    w = int(bbox_str[2])
                    h = bbox_str[3]
                    h = int(h[0:-1])
                    bbox_init = (x,y,w,h)
                    print(bbox_init)
                    
                    # Create output video
                    output_folder = 'outputVideos/' + video_type + '/'
                    video_output_file_name = video_input_file_name + '_' + tracker_type + '_' + str(num_cible + 1)
                    video_output_file_extension = ".avi"
                    video_out = cv2.VideoWriter(output_folder+video_output_file_name+video_output_file_extension, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
                        
                    # Exit if video output is not opened
                    if not video_out.isOpened():
                        print("Could not write video output")
                        sys.exit()
                
                    # Read first frame
                    ok, frame = video.read()
                    if not ok:
                        print('Cannot read video file')
                        sys.exit()
    
                    # Draw bounding box
                    if ok and verifBbox(bbox_init,0,width,0,height,0,width,0,height):
                        drawRectangle(frame,bbox_init)

                    # Initialize tracker with first frame and bounding box
                    tracker = createTracker(tracker_type)
                    ok = tracker.init(frame, bbox_init)

                    # Initialize execution timers
                    tick_freq = cv2.getTickFrequency()
                    frame_count = 0
                    total_exec_time = 0
                    
                    # Initialize fail timers
                    fail = False
                    fail_time = 0
                    frames_computed_per_second = 0
                    
                    while True:
                        # Read a new frame
                        ok, frame = video.read()
                        if not ok:
                            break
                        
                        # Start timer
                        timer = cv2.getTickCount()

                        # Update tracker
                        ok, bbox = tracker.update(frame)
                        
                        # Calculate execution time
                        exec_time = max((cv2.getTickCount() - timer) / tick_freq,0)

                        # Draw bounding box
                        if ok and verifBbox(bbox,0,width,0,height,0,width,0,height):
                            # Tracking success and bbox in the image
                            drawRectangle(frame,bbox) 
                        else :
                            # Tracking failure
                            if not fail:
                                fail_time = frame_count
                                frames_computed_per_second = ceil(100*frame_count/total_exec_time)/100
                            fail = True
                            drawText(frame, "Tracking failure detected", (100,90), (0,0,255))

                        # Display tracker type on frame
                        drawText(frame, "Tracker " + tracker_type, (100,40))
                        
                        # Display FPS on frame
                        frame_count += 1
                        total_exec_time += exec_time
                        drawText(frame, "Frames computed per s : " + str(ceil(100*frame_count/total_exec_time)/100), (100,70))

                        # Uncomment to display result
                        #cv2.imshow("Tracking", frame)
                        
                        # Register frame results
                        video_out.write(frame)

                        # Exit if ESC pressed
                        k = cv2.waitKey(1) & 0xff
                        if k == 27 : break

                    video.release()
                    video_out.release()
                    
                    if not fail:
                        frames_computed_per_second = ceil(100*frame_count/total_exec_time)/100
                    
                    assert num_test == data['Test'][num_cible]  
                    data['Output'][num_cible] = video_output_file_name + video_output_file_extension
                    data['FPS'][num_cible] = fps
                    data['Total frames'][num_cible] = total_frame
                    data['Initialisation'][num_cible] = video_type + " " + str(num_cible+1)
                    data['Initialisation scale'][num_cible] = str(bbox_init)
                    data['Frames Computed per Second'][num_cible] = frames_computed_per_second
                    data['Tracking Failure'][num_cible] = fail
                    if fail :
                        data['Frames before'][num_cible] = fail_time
                    
                    num_test += 1

                # Register data
                data_folder = 'dataFiles/test/' + video_type + '/'
                data_frame = pandas.DataFrame(data,index=[i for i in range(N)])
                print(data_frame)
                data_frame.to_excel(data_folder+video_input_file_name + '_' + tracker_type+'.xlsx', sheet_name='data',index=[i for i in range(N)])