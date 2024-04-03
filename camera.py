import cv2
import numpy as np
from aravis import Camera
from evasdk import Eva

arm_hostname = "evatrendylimashifterpt410"
arm_ip = "144.32.152.105"
token = "1462980d67d58cb7eaabe8790d866609eb97fd2c"

pixcel_len = 30/850

eva = Eva(arm_ip, token)
with eva.lock():
  eva.control_wait_for_ready()
  eva.control_go_to([0, 1, -2.5, 0, -1.6, 0])

from json import dumps
toolpaths = eva.toolpaths_list()
outToolpaths = []
for toolpathItem in toolpaths:
  toolpath = eva.toolpaths_retrieve(toolpathItem['id'])
  outToolpaths.append(toolpath)
print(dumps(outToolpaths))

toolpath = {
  "metadata": {
      "version": 2,
      "payload": 0,
      "default_max_speed": 1.05,
      "next_label_id": 5,
      "analog_modes": {"i0": "voltage", "i1": "voltage", "o0": "voltage", "o1": "voltage"},
  },
  "waypoints": [
      {"joints": [-0.68147224, 0.3648368, -1.0703622, 9.354615e-05, -2.4358354, -0.6813218], "label_id": 3},
      {"joints": [-0.6350288, 0.25192022, -1.0664424, 0.030407501, -2.2955494, -0.615318], "label_id": 2},
      {"joints": [-0.13414459, 0.5361486, -1.280493, -6.992453e-08, -2.3972468, -0.13414553], "label_id": 1},
      {"joints": [-0.4103904, 0.33332264, -1.5417944, -5.380291e-06, -1.9328799, -0.41031334], "label_id": 4},
  ],
  "timeline": [
      {"type": "home", "waypoint_id": 2},
      {"type": "trajectory", "trajectory": "joint_space", "waypoint_id": 1},
      {"type": "trajectory", "trajectory": "joint_space", "waypoint_id": 0},
      {"type": "trajectory", "trajectory": "joint_space", "waypoint_id": 2},
  ],
}

with eva.lock():
  eva.control_wait_for_ready()
  eva.toolpaths_use(toolpath)
  eva.control_home()
  eva.control_run(loop=1)

with eva.lock():
  eva.control_reset_errors()

with eva.lock():
   eva.control_wait_for_ready()
   eva.gpio_set('ee_d1', False)
   eva.gpio_set('ee_d0', True)

with eva.lock():
   eva.control_wait_for_ready()
   eva.gpio_set('ee_d0', False)
   eva.gpio_set('ee_d1', True)

with eva.lock():
   ee_d0_state = eva.gpio_get('ee_d0', 'output')
   ee_d1_state = eva.gpio_get('ee_d1', 'output')

cam = Camera()
cam.set_feature("Width", 1936)
cam.set_feature("Height", 1216)
cam.set_frame_rate(10)
cam.set_exposure_time(100000)
cam.set_pixel_format_from_string('BayerRG8')

try:
   cam.start_acquisition_continuous()
   print("Camera On")
   cv2.namedWindow('capture', flags=0)
   count = 0

   while True:
      # Capture frame-by-frame
      #ret, frame = cam.read()
      
      frame = cam.get_image()

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
         print("Center coordinates: ({}, {})".format(cx, cy))
        
         cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

         # Draw circle at centroid location
         cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

         # Display frame with centroid
         cv2.imshow('Frame with Centroid', frame)
         cv2.waitKey(1)

except KeyboardInterrupt:
   print("Exiting...")
finally:
   cam.stop_acquisition()
   cam.shutdown()
   print("Camera Off")

