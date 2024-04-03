#importing libraries
import cv2
import numpy as np
from evasdk import Eva

#conecting for arm 
arm_hostname = "evatrendylimashifterpt410"
arm_ip = "144.32.152.105"
token = "1462980d67d58cb7eaabe8790d866609eb97fd2c"

eva = Eva(arm_ip, token)

#home position
with eva.lock():
  eva.control_wait_for_ready()
  eva.control_go_to([0, 1, -2.5, 0, -1.6, 0])

# Define video capture device
cap = cv2.VideoCapture(0)

# continuous loop 
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
	# Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10,255,255])
    
    # Create mask of pixels within color range
    mask = cv2.inRange(hsv, lower_red, upper_red)
  
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Determine centroid of largest contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largest_contour)
        cx = int(x + w/2)
        cy = int(y + h/2) 

        Xo = -((cx - (1936/2))*0.01333/100)
        Yo = -((cy - (1216/2))*0.01333/100)

        print("Center coordinates: ({}, {})".format(cx, cy))
        print('\n********************************************\n')
        print("Center coordinates: ({}, {})".format(Xo, Yo))
      
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        # Draw circle at centroid location
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # Display frame with centroid
    cv2.imshow('Frame with Centroid', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture device and close window
cap.release()
cv2.destroyAllWindows()