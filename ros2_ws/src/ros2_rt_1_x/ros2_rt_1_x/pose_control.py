import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from geometry_msgs.msg import Pose

import time
import pygame
import sys

# MOVEMENT LIMITS
X_MIN = -0.3
X_MAX = 0.3
Y_MIN = 0.4
Y_MAX = 0.7
Z_MIN = -15.0
Z_MAX = 0.7
O_X_MIN = 45.0
O_X_MAX = 135.0
O_Y_MIN = 0.0
O_Y_MAX = 90.0
O_Z_MIN = 0.0
O_Z_MAX = 90.0
GRIP_MIN = 0.02
GRIP_MAX = 0.08

class PoseControl(Node):
    def __init__(self):
        super().__init__('pose_control')
        self.pose_publisher = self.create_publisher(Pose, 'target_pose', 10)
        self.grip_publisher = self.create_publisher(Float32, 'target_grip', 10)

        self.current_pose = Pose()
        self.current_grip = Float32()

        self.current_pose.position.x = 0.0
        self.current_pose.position.y = 0.5
        self.current_pose.position.z = 0.5
        self.current_pose.orientation.x = 90.0
        self.current_pose.orientation.y = 0.0
        self.current_pose.orientation.z = 0.0
        self.current_pose.orientation.w = 1.0
        self.current_grip.data = 0.05

        pygame.init()
        self.screen = pygame.display.set_mode((100, 100))
        self.timer = self.create_timer(1, self.main)
        self.joysticks = {}

    def enforce_limits(self):
        self.current_pose.position.x = max(X_MIN, min(X_MAX, self.current_pose.position.x))
        self.current_pose.position.y = max(Y_MIN, min(Y_MAX, self.current_pose.position.y))
        self.current_pose.position.z = max(Z_MIN, min(Z_MAX, self.current_pose.position.z))
        self.current_pose.orientation.x = max(O_X_MIN, min(O_X_MAX, self.current_pose.orientation.x))
        self.current_pose.orientation.y = max(O_Y_MIN, min(O_Y_MAX, self.current_pose.orientation.y))
        self.current_pose.orientation.z = max(O_Z_MIN, min(O_Z_MAX, self.current_pose.orientation.z))
        self.current_grip.data = max(GRIP_MIN, min(GRIP_MAX, self.current_grip.data))

    def main(self):
        self.timer.cancel()
        joy = self.joysticks.get(0)

        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.JOYDEVICEADDED:
                    # This event will be generated when the program starts for every
                    # joystick, filling up the list without needing to create them manually.
                    joy = pygame.joystick.Joystick(event.device_index)
                    self.joysticks[joy.get_instance_id()] = joy
                    print(f"Joystick {joy.get_instance_id()} connencted")

                # if event.type == pygame.JOYBUTTONDOWN:
                #     print("Joystick button pressed.")
                #     if event.button == 0:
                #         joystick = self.joysticks[event.instance_id]
                #         if joystick.rumble(0, 0.7, 500):
                #             print(f"Rumble effect played on joystick {event.instance_id}")

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.current_pose.position.x += 0.1
                    elif event.key == pygame.K_d:
                        self.current_pose.position.x -= 0.1
                    elif event.key == pygame.K_w:
                        self.current_pose.position.y -= 0.1
                    elif event.key == pygame.K_s:
                        self.current_pose.position.y += 0.1
                    elif event.key == pygame.K_UP:
                        self.current_pose.position.z += 0.1
                    elif event.key == pygame.K_DOWN:
                        self.current_pose.position.z -= 0.1

                    self.pose_publisher.publish(self.current_pose)
                    self.get_logger().info(f'Published target pose: {self.current_pose}')

            move = False
            if joy:
                # move grip
                if joy.get_button(6) == 1:
                    move = True
                    self.current_grip.data += 0.01
                elif joy.get_button(7) == 1:
                    move = True
                    self.current_grip.data -= 0.01

                # move x,y
                x = joy.get_axis(0)
                y = joy.get_axis(1)
                if abs(x) > 0.05 or abs(y) > 0.05:
                    move = True
                    self.current_pose.position.x += x * -0.01
                    self.current_pose.position.y += y * 0.01

                # move z
                if joy.get_button(4) == 1:
                    move = True
                    self.current_pose.position.z += 0.01
                elif joy.get_button(5) == 1:
                    move = True
                    self.current_pose.position.z -= 0.01

                # move pitch
                pitch = joy.get_axis(2)
                if abs(pitch) > 0.05:
                    move = True
                    self.current_pose.orientation.y += pitch * 1

                # move yaw
                yaw = joy.get_axis(3)
                if abs(yaw) > 0.05:
                    move = True
                    self.current_pose.orientation.x += yaw * 1



                # z = joy.get_axis(2)
                # o_x = joy.get_axis(3)
                # o_y = joy.get_axis(4)
                # o_z = joy.get_axis(5)
                # grip = joy.get_axis(6)

                

                
                # self.current_pose.position.z += z * 0.1
                # self.current_pose.orientation.x += o_x * 0.1
                # self.current_pose.orientation.y += o_y * 0.1
                # self.current_pose.orientation.z += o_z * 0.1
                # self.current_grip.data += grip * 0.1

                if move:
                    self.enforce_limits()
                    self.pose_publisher.publish(self.current_pose)
                    self.grip_publisher.publish(self.current_grip)
                    self.get_logger().info(f'Published target pose: {self.current_pose}')
                    self.get_logger().info(f'Published target grip: {self.current_grip}')
                    move = False

                time.sleep(0.1)



            

def main(args=None):
    rclpy.init(args=args)

    pose_control = PoseControl()

    rclpy.spin(pose_control)

    pose_control.destroy_node()
    rclpy.shutdown()