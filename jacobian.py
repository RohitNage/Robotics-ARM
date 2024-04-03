from sympy import Matrix, Symbol, symbols, solveset, solve, simplify, S, diff, det, erf, log, sqrt, pi, sin, cos, tan, asin, acos, atan2, init_printing
import numpy
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from sympy import Matrix, Symbol, symbols, solveset, solve, simplify, S, diff, det, erf, log, sqrt, pi, sin, cos, tan, asin, acos, atan2, init_printing

#from evasdk import Eva

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

theta1,theta2,theta3,theta4,theta5,theta6 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6')
theta = Matrix([theta1,theta2,theta3,theta4,theta5,theta6])

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

J = p.jacobian(theta)

theta_i = Matrix([0, 1, -2.5, 0, -1.6, 0])

p_i = p.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()

p_f = p_i + Matrix([ 0.5, 0.5, 0.5 ])
#p_f = Matrix([0.3, 0.1, 0.1]) # this the place where you will substitute the x and y, z is constant)

dp = p_f - p_i
dp_threshold = 0.01
while dp.norm() > dp_threshold:
   '''
   print("step “,step,”:\n θ[",theta_i,"]\n p[",p_i,"]")
   '''
   step_size = 0.05
   theta_max_step = 0.2

   dp_step = dp * step_size / dp.norm()
   J = p.jacobian(theta)
   J_i = J.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()

   J_inv = J_i.pinv()
   dtheta = J_inv * dp_step

   theta_i = theta_i + numpy.clip(dtheta,-1*theta_max_step,theta_max_step)

   p_i = p.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()

   dp = p_f - p_i

   pp0sub = p0.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
   p1sub = p1.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
   p2sub = p2.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
   p3sub = p3.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
   p4sub = p4.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
   p5sub = p5.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()
   p6sub = p6.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]}).evalf()

   soa = numpy.array([pp0sub,p1sub,p2sub,p3sub,p4sub,p5sub,p6sub])

   X, Y, Z, W = zip(*soa)
   
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
   
   ax.view_init(elev=45, azim=45)
   ax.plot3D(X,Y,Z, 'blue', marker="o")
   
   matplotlib.pyplot.draw()
   matplotlib.pyplot.show()
   matplotlib.pyplot.pause(0.1)

print('Final Theta Angles is :',theta_i.evalf())
