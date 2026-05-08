from trash_bot.CV.controller.mpc_controller import MPCController
import time
import rclpy
from geometry_msgs.msg import Twist
from trash_bot.arduino.red import main as red_on
from trash_bot.arduino.green import main as green_on
from trash_bot.arduino.blue import main as blue_on
from trash_bot.arduino.turn180 import main as turn_camera_180
from trash_bot.arduino.turn0 import main as turn_camera_0
from trash_bot.CV.trashbot_dir.ros_classify import classify_item
from trash_bot.CV.trashbot_dir.ros_hand_detect import detect_hand
import math
from nav_msgs.msg import Odometry
from trash_bot.CV.controller.object_dector import run_one_shot_obstacle_scan

TIMER_DELAY_SEC = 5.0

def robot_pose_estimation_callback(msg):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - (2.0 * (q.y * q.y + q.z * q.z))
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (p.x, p.y, yaw)

def get_cur_pose():
    pose = {"value": None}

    def odom_callback(msg):
        pose["value"] = robot_pose_estimation_callback(msg)

    node = rclpy.create_node('pose_estimation_node')
    sub = node.create_subscription(Odometry, '/odom', odom_callback, 10)

    while rclpy.ok() and pose["value"] is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_subscription(sub)
    node.destroy_node()
    return pose["value"]

def camera_to_world(xyz_camera, robot_pose):
    cam_x, _, cam_z = xyz_camera
    robot_x, robot_y, robot_yaw = robot_pose

    forward_m = cam_z
    right_m = -cam_x

    # world_x = robot_x + cam_z
    # world_y = robot_y - cam_x
    world_x = robot_x + forward_m * math.cos(robot_yaw) - right_m * math.sin(robot_yaw)
    world_y = robot_y + forward_m * math.sin(robot_yaw) + right_m * math.cos(robot_yaw)

    # MPC goal is [x, y, theta] in odom/world coordinates.
    # Keep the current yaw at the hand target to avoid adding an arbitrary final turn.
    return (world_x, world_y, robot_yaw)


def main():
    print("This is the trashbot state module.")
    goal = [0.0, 0.0, 0.0]

    state = "search"
    
    while True:
        match state:
            case "search":
                print("Searching for hand signal...")
                result = detect_hand()

                print("Hand result:", f"found={result.found}", f"xyz={result.xyz_m}")

                if result.found:
                    print(result.xyz_m)
                    # print(result.direction_x, result.direction_y)

                    if result.xyz_m is not None:
                        if not rclpy.ok():
                            rclpy.init()
                        robot_pose = get_cur_pose()
                        goal = camera_to_world(result.xyz_m, robot_pose)
                        rclpy.shutdown()
                        print(f"camera xyz (m): {result.xyz_m}")
                        print(f"robot pose (x, y, yaw): {robot_pose}")
                        print(f"world goal (x, y, yaw): {goal}")
                        state = "approach"
                    else:
                        print("Hand detected but xyz is unavailable, staying in search.")
                        state = "search"
                else:
                    state = "search"

            case "approach":
                print("Approaching the hand signal...")
                if not rclpy.ok():
                    rclpy.init()
                initial_pose = get_cur_pose()
                rclpy.shutdown()
                home_goal = [initial_pose[0], initial_pose[1], initial_pose[2]]
                #goal = (initial_pose[0], initial_pose[1] + 0.2, initial_pose[2])
                print(f"Initial pose saved as home goal: {home_goal}")
                if not rclpy.ok():
                   rclpy.init()
                print(f"Goal: {goal}")
                if goal[0] >= 0.0 and goal[1] >= 0.0:
                    goal = (goal[0] - .3, goal[1]-.3, goal[2])
                elif goal[0] >= 0.0 and goal[1] < 0.0:
                    goal = (goal[0] - .3, goal[1]+.3, goal[2])
                elif goal[0] < 0.0 and goal[1] >= 0.0:
                    goal = (goal[0] + .3, goal[1]-.3, goal[2])
                elif goal[0] < 0.0 and goal[1] < 0.0:
                    goal = (goal[0] + .3, goal[1]+.3, goal[2])
                print(f"Adjusted goal for better approach: {goal}")
                goal = (goal[0], goal[1], initial_pose[2])
                obstacles = run_one_shot_obstacle_scan()
                print(f"Obstacles detected: {len(obstacles)}")
                node = MPCController(goal=goal, initial_pose=initial_pose, obstacles=obstacles)
                while rclpy.ok() and not node.goal_reached:
                    rclpy.spin_once(node, timeout_sec=0.1)
                node.control_pub.publish(Twist())
                node.destroy_node()
                rclpy.shutdown()
                state = "classify"
            case "classify":
                print("Classifying the trash...")
                time.sleep(TIMER_DELAY_SEC)
                if not rclpy.ok():
                   rclpy.init()
                node = MPCController(goal=(goal[0], goal[1], math.atan2(math.sin(initial_pose[2]+math.pi), math.cos(initial_pose[2]+math.pi))), initial_pose=initial_pose, obstacles=obstacles)
                while rclpy.ok() and not node.goal_reached:
                    rclpy.spin_once(node, timeout_sec=0.1)
                node.control_pub.publish(Twist())
                node.destroy_node()
                rclpy.shutdown()
                turn_camera_180()
                label = classify_item()
                turn_camera_0()
                if label == "Trash":
                    red_on()
                elif label == "Compost":
                    green_on()
                elif label == "Recycling":
                    blue_on()
                if not rclpy.ok():
                   rclpy.init()
                node = MPCController(goal=(goal[0], goal[1], initial_pose[2]), initial_pose=initial_pose, obstacles=obstacles)
                while rclpy.ok() and not node.goal_reached:
                    rclpy.spin_once(node, timeout_sec=0.1)
                node.control_pub.publish(Twist())
                node.destroy_node()
                rclpy.shutdown()
                time.sleep(TIMER_DELAY_SEC)
                state = "retreat"
            case "retreat":
                print("Returning to the starting point...")
                print(f"Waiting for {TIMER_DELAY_SEC} seconds before retreating...")
                time.sleep(TIMER_DELAY_SEC)
                if not rclpy.ok():
                   rclpy.init()
                node = MPCController(goal=(goal[0], goal[1], math.atan2(math.sin(initial_pose[2]+math.pi), math.cos(initial_pose[2]+math.pi))), initial_pose=initial_pose, obstacles=obstacles)
                while rclpy.ok() and not node.goal_reached:
                    rclpy.spin_once(node, timeout_sec=0.1)
                node.control_pub.publish(Twist())
                node.destroy_node()
                rclpy.shutdown()
                if not rclpy.ok():
                   rclpy.init()
                obstacles = run_one_shot_obstacle_scan()
                print(f"Obstacles detected: {len(obstacles)}")
                node = MPCController(goal=home_goal, initial_pose=goal, obstacles=obstacles)
                while rclpy.ok() and not node.goal_reached:
                    rclpy.spin_once(node, timeout_sec=0.1)
                node.control_pub.publish(Twist())
                node.destroy_node()
                rclpy.shutdown()
                state = "search"
            case _:
                print("Unknown state.")
        


if __name__ == "__main__":
    main()