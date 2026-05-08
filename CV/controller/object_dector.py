import json
import math
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class OneShotObstacleScan(Node):
    def __init__(self):
        super().__init__("one_shot_obstacle_scan")

        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None
        self.latest_scan = None
        self.final_obs = None

        self.output_file = Path("controller/obstacles.json")

        self.min_range_m = 0.25
        self.max_range_m = 5.0
        self.front_angle_deg = 45.0
        self.lateral_limit_m = 0.70
        self.max_obstacles = 1
        self.obstacle_radius = 0.35

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/trashbot/depth_scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        self.get_logger().info("Waiting for /odom and /trashbot/depth_scan...")

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg 
    
    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def process_once(self):
        obstacles = []
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment

        for i, range_m in enumerate(self.latest_scan.ranges):
            if not math.isfinite(range_m):
                continue

            if range_m < self.min_range_m or range_m > self.max_range_m:
                continue
            
            angle_deg = angle_min + i * angle_increment
            if abs(math.degrees(angle_deg)) > self.front_angle_deg:
                continue
            
            x_robot = range_m * math.cos(angle_deg)
            y_robot = range_m * math.sin(angle_deg)

            if abs(y_robot) > self.lateral_limit_m:
                continue
            
            x_world = self.robot_x + x_robot * math.cos(self.robot_yaw) - y_robot * math.sin(self.robot_yaw)
            y_world = self.robot_y + x_robot * math.sin(self.robot_yaw) + y_robot * math.cos(self.robot_yaw)

            obstacles.append({
                "x": x_world,
                "y": y_world,
                "radius": self.obstacle_radius,
            })

            # if len(obstacles) >= self.max_obstacles:
            #     break
        if not obstacles:
            return []

        final_obs = [obstacles[0]]
        for i, obstacle in enumerate(obstacles):
            temp_dist = self.distance(obstacle["x"], obstacle["y"], self.robot_x, self.robot_y)
            if temp_dist < self.distance(final_obs[0]["x"], final_obs[0]["y"], self.robot_x, self.robot_y):
                final_obs[0] = obstacle

        with open(self.output_file, "w") as f:
            json.dump({"obstacles": final_obs}, f, indent=2, allow_nan=False)
        self.final_obs = final_obs
        self.get_logger().info(f"Saved {len(final_obs)} obstacles to {self.output_file}")
        return final_obs


def run_one_shot_obstacle_scan():
    own_context = not rclpy.ok()
    if own_context:
        rclpy.init()

    node = OneShotObstacleScan()
    try:
        while rclpy.ok() and (
            node.robot_x is None or node.robot_y is None or node.robot_yaw is None or node.latest_scan is None
        ):
            rclpy.spin_once(node, timeout_sec=0.1)

        if rclpy.ok():
            obstacles = node.process_once()
        

    finally:
        node.destroy_node()
        if own_context and rclpy.ok():
            rclpy.shutdown()

    return obstacles
    
def main(args=None):
    obstacles = run_one_shot_obstacle_scan()
    print(f"Detected {len(obstacles)} obstacles")

if __name__ == "__main__":
    main()
    