#imports various modules
from sympy import Matrix, Symbol, symbols, solveset, solve, simplify, S, diff, det, erf, log, sqrt, pi, sin, cos, tan, asin, acos, atan2, init_printing
import numpy
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from sympy import Matrix, Symbol, symbols, solveset, solve, simplify, S, diff, det, erf, log, sqrt, pi, sin, cos, tan, asin, acos, atan2, init_printing
from evasdk import Eva
import time
import cv2
from aravis import Camera

#defining angles
theta1,theta2,theta3,theta4,theta5,theta6 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6')
theta = Matrix([theta1,theta2,theta3,theta4,theta5,theta6])


#define starting position
def starting_pos():
    with eva.lock():
        eva.control_wait_for_ready()
        eva.control_go_to([0,1,-2.5,0,-1.6,0])
        print("the robot arm is in initial position")

#define final position
def end_position():
    with eva.lock():
        eva.control_wait_for_ready()
        eva.control_go_to([0,1,-2.5,0,-1.6,0])
        print("the robot arm is in final position")
    
#this define the transformation matrices for translations and rotations about the x, y, and z axes respectively.
def T(x, y, z):
   T_xyz = Matrix([[1,         0,          0,          x],
                   [0,         1,          0,          y],
                   [0,         0,          1,          z],
                   [0,         0,          0,          1]])
   return T_xyz

def Rx(roll):
   R_x = Matrix([[1,         0,          0, 0],
                 [0, cos(roll), -sin(roll), 0],
                 [0, sin(roll),  cos(roll), 0],
                 [0,         0,          0, 1]])
   return R_x

def Ry(pitch):
   R_y = Matrix([[ cos(pitch), 0, sin(pitch), 0],
                 [          0, 1,          0, 0],
                 [-sin(pitch), 0, cos(pitch), 0],
                 [          0, 0,          0, 1]])
   return R_y

def Rz(yaw):
   R_z = Matrix([[cos(yaw),-sin(yaw), 0, 0],
                 [sin(yaw), cos(yaw), 0, 0],
                 [       0,        0, 1, 0],
                 [       0,        0, 0, 1]])
   return R_z

#define the value of thresold, step_size, thetha_max_step
dp_threshold = 0.01
step_size = 0.05
theta_max_step = 0.2

# Define transforms to each joint
T1 = Ry(-pi/2) * T(0.187, 0, 0) * Rx(theta1)
T2 = T1 * T(0.096, 0, 0) * Rz(theta2)
T3 = T2 * T(0.205, 0, 0) * Rz(theta3)
T4 = T3 * T(0.124, 0, 0) * Rx(theta4)
T5 = T4 * T(0.167, 0, 0) * Rz(theta5)
T6 = T5 * T(0.104, 0, 0) * Rx(theta6)

# Find joint positions in space
p0 = Matrix([0,0,0,1])
p1 = T1 * p0
p2 = T2 * p0
p3 = T3 * p0
p4 = T4 * p0
p5 = T5 * p0
p6 = T6 * p0

p = Matrix([p6[0], p6[1], p6[2]]) # coordinates of arm tip

theta_i = Matrix([0,1,-2.5,0,-1.6,0]) # home position

#defining function for final moment 
def final_movement():
    eva = Eva(arm_ip, token)
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

#define jacobian transform matrix and calcualtion
def jacobian_joints(c_x, c_y):
    J = p.jacobian(theta)
    c_z = -0.07
    x_position = -((c_x - (1936/2))*0.01333/100)
    y_position = -((c_y - (1216/2))*0.01333/100)
    theta_i = Matrix([0,1,-2.5,0,-1.6,0])
    
    p_i = p.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
     
    p_f = p_i + Matrix([x_position - 0.03 ,y_position - 0.02, c_z])
   
    dp = p_f - p_i
    while dp.norm() > dp_threshold:
       
        dp_step = dp * step_size / dp.norm() 
        J = p.jacobian(theta)
        J_i = J.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
        J_inv = J_i.pinv()
        dtheta = J_inv * dp_step
        theta_i = theta_i + numpy.clip(dtheta,-1*theta_max_step,theta_max_step)
        p_i = p.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
        dp = p_f - p_i
        p0sub = p0.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
        p1sub = p1.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
        p2sub = p2.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
        p3sub = p3.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
        p4sub = p4.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
        p5sub = p5.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
        p6sub = p6.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
        
        soa = numpy.array([p0sub,p1sub,p2sub,p3sub,p4sub,p5sub,p6sub])
        X, Y, Z, W  = zip(*soa)
        X = numpy.array(X)
        Y = numpy.array(Y)
        Z = numpy.array(Z)
        W = numpy.array(W)
        X = numpy.ndarray.flatten(X)
        Y = numpy.ndarray.flatten(Y)
        Z = numpy.ndarray.flatten(Z)
        W = numpy.ndarray.flatten(W)
        fig = matplotlib.pyplot.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([0, 0.5])
        ax.view_init(elev=0, azim=180)
        ax.plot3D(X,Y,Z, 'blue', marker="o")
        matplotlib.pyplot.draw()
        matplotlib.pyplot.show()
        matplotlib.pyplot.pause(0.1)
        theta_1 = float(theta_i[0])
        
        theta_2 = float(theta_i[1])
        
        theta_3 = float(theta_i[2])
        
        theta_4 = float(theta_i[3])
        
        theta_5 = float(theta_i[4])
        
        theta_6 = float(theta_i[5])
        
        
    print('\n\nFinal Joint Angles in Radians:\n', theta_i.evalf())    
    with eva.lock():
       eva.control_wait_for_ready()
       print("The robot is moving to the desired position")
       eva.control_go_to([theta_1,theta_2,theta_3,theta_4,theta_5,theta_6])

#defining camera function
def camera():
       cam.start_acquisition_continuous()
       frame = cam.pop_frame()
       
       if not 0 in frame.shape:
          
          bgr = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2BGR)
          hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
          lower_green = numpy.array([40, 25, 25])
          upper_green = numpy.array([70, 255, 255])
          mask = cv2.inRange(hsv, lower_green, upper_green)        
          contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
          if len(contours) > 0:
              largest_contour = max(contours, key=cv2.contourArea)
              x,y,w,h = cv2.boundingRect(largest_contour)
              cx = int(x + w/2)
              cy = int(y + h/2)
              print("Center coordinates: ({}, {})".format(cx, cy))
              cv2.rectangle(bgr, (x,y), (x+w, y+h), (0,0, 255), 2)
              cv2.circle(bgr, (cx, cy), 5, (0, 0, 255), -1)
              cam.stop_acquisition()
              cv2.imshow("Frame", bgr) 
              return cx, cy
           
#main function
if __name__ == '__main__':
    
    arm_hostname = "evatrendylimashifterpt410"
    arm_ip = "144.32.152.105"
    token = "1462980d67d58cb7eaabe8790d866609eb97fd2c"
    
    camera_hostname = "evacctv02"
    camera_ip = "144.32.152.10"

    cam = Camera('S1188411')
    cam.set_feature("Width", 1936)
    cam.set_feature("Height", 1216)
    cam.set_frame_rate(10)
    cam.set_exposure_time(100000)
    cam.set_pixel_format_from_string('BayerRG8')
    
    eva = Eva(arm_ip, token)
    starting_pos()
    a,b = camera()
    jacobian_joints(a,b)
    time.sleep(0.1)
    
    with eva.lock():
        eva.control_wait_for_ready()
        eva.gpio_set('ee_d1', False)
        eva.gpio_set('ee_d0', True)
    time.sleep(0.5)
    with eva.lock():
        eva.control_wait_for_ready()
        eva.gpio_set('ee_d1', False)
        eva.gpio_set('ee_d0', True)
    #final_movement()
    #end_position()
               
        
          



    
    


