import cv2
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

def drawText(frame, txt, location, color=(50, 170, 50)):
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

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[6]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.legacy.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.legacy.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.legacy.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.legacy.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.legacy.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.legacy.TrackerMOSSE_create()
        if tracker_type == 'CSRT':
            tracker = cv2.legacy.TrackerCSRT_create()

    # Read video
    input_folder = "inputVideos/"
    video_input_file_name = "bateau_1"
    video_input_file_extension = ".mp4"
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
    
    # Create output video
    output_folder = "outputVideos/"
    video_output_file_name = video_input_file_name + "_" + tracker_type
    video_output_extension = ".avi"
    video_out = cv2.VideoWriter(output_folder+video_output_file_name+video_output_extension, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
    
    # Exit if video output is not opened
    if not video_out.isOpened():
        print("Could not write video output")
        sys.exit()

    # Read first frame
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Define an initial bounding box
    bbox = (421, 501, 118, 184)
    
    # Uncomment the line below to select a different bounding box
    #bbox = cv2.selectROI(frame, False)
    
    # Draw bounding box
    if ok and verifBbox(bbox,0,width,0,height,0,width,0,height):
        print(bbox)
        drawRectangle(frame,bbox)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

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
        frame_count += 1
        
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate execution time
        exec_time = (cv2.getTickCount() - timer) / tick_freq
        total_exec_time += exec_time
        
        # Draw bounding box
        if ok and verifBbox(bbox,0,width,0,height,0,width,0,height):
            # Tracking success and bbox in the image
            drawRectangle(frame,bbox)
        else :
            # Tracking failure
            if not fail:
                fail_time = frame_count
                frames_computed_per_second = int((frame_count-1)/(total_exec_time-exec_time))
            fail = True
            drawText(frame, "Tracking failure detected", (100,90), (0,0,255))

        # Display tracker type on frame
        drawText(frame, "Tracker " + tracker_type, (100,40))
    
        # Display FPS on frame
        drawText(frame, "Frames computed per s : " + str(int(frame_count/total_exec_time)), (100,70))

        # Uncomment to display result
        #cv2.imshow("Tracking", frame)
        
        # Register frame results
        video_out.write(frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

print(fail_time)
print(total_frame)
print(int(frames_computed_per_second))

video.release()
video_out.release()